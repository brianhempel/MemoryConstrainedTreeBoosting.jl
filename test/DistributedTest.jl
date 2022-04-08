
# echo "import MPI\nMPI.install_mpiexecjl()" | julia --project=..
#
# $ mpiexecjl -n 3 julia --project=.. DistributedTest.jl
#
# This file is the MPI version of Test.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

import Random
using MPI
using MemoryConstrainedTreeBoosting

using MPI
MPI.Init()
comm       = MPI.COMM_WORLD
root       = 0
rank       = MPI.Comm_rank(comm)
rank_count = MPI.Comm_size(comm)

print("Hi from rank $(MPI.Comm_rank(comm)) of $(MPI.Comm_size(comm))!\n")
MPI.Barrier(comm)

function chunk_range(chunk_i, n_chunks, array_len)
  start = div((chunk_i-1) * array_len, n_chunks) + 1
  stop  = div( chunk_i    * array_len, n_chunks)
  start:stop
end

Random.seed!(123456)

# 10x3 array
X_full = Float32[0.314421   0.622812  0.758382
            0.592915   0.799313  0.82079
            0.121827   0.946241  0.00250338
            0.248926   0.44318   0.375335
            0.0302476  0.365399  0.18079
            0.0598305  0.463519  0.0609047
            0.510357   0.488909  0.655259
            0.358932   0.98549   0.472883
            0.443199   0.4372    0.476574
            0.195279   0.576752  0.436448]

y_full = Float32[0.0
              1.0
              0.0
              0.0
              0.0
              0.0
              1.0
              1.0
              1.0
              1.0]

weights_full = Float32[10.0
                  0.1
                  0.1
                  1.0
                  1.0
                  1.0
                  1.0
                  1.0
                  10.0
                  1.0]

bin_splits = rank == root ? prepare_bin_splits(X_full, bin_count = 4) : nothing
bin_splits = MPI.bcast(bin_splits, root, comm)
Random.seed!(123456 + rank)

# Simulate data being on different nodes
my_chunk_range = chunk_range(rank+1,rank_count,10)
X       = X_full[my_chunk_range, :]
y       = y_full[my_chunk_range]
weights = weights_full[my_chunk_range]

my_validation_range = chunk_range(rank+1,rank_count,3)
validation_X       = X_full[1:3, :][my_validation_range, :]
validation_y       = y_full[1:3][my_validation_range]
validation_weights = weights_full[1:3][my_validation_range]

X_binned            = apply_bins(X, bin_splits)
validation_X_binned = apply_bins(validation_X, bin_splits)

make_callback() = make_callback_to_track_validation_loss(
  validation_X_binned,
  validation_y;
  mpi_comm = comm,
  validation_weights = validation_weights
)

trees = train_on_binned(
  X_binned, y,
  weights                 = weights,
  iteration_callback      = make_callback(),
  iteration_count         = 20,
  min_data_weight_in_leaf = 2.0,
  learning_rate           = 0.3,
  bagging_temperature     = 0.0, # otherwise, non-deterministic
  mpi_comm                = comm,
);

MPI.Barrier(comm)
if rank == root
  expected_ŷ_full = Float32[0.030087346, 0.9722849, 0.867269, 0.07883832, 0.07883832, 0.07883832, 0.9722849, 0.867269, 0.9722849, 0.867269]

  println()
  ŷ_full = predict(X_full, bin_splits, trees)
  println("           y: $(y_full)")
  println("The following two should be the same:")
  println("           ŷ: $(ŷ_full)")
  println("  expected ŷ: $(expected_ŷ_full)")
  println()

  @assert ŷ_full == expected_ŷ_full # this will segfault MPI if it fails. oh well
end

rank == root && println("Testing early stopping...")

# Dirty the validation labels
# validation_y = map(y -> rand() < 0.2 ? 1f0 - y : y, validation_y)

max_iterations_without_improvement = 10
make_callback() = make_callback_to_track_validation_loss(
  validation_X_binned,
  validation_y;
  mpi_comm = comm, # May deadlock if you forget this, because some procs will stop before others
  validation_weights = validation_weights,
  max_iterations_without_improvement = max_iterations_without_improvement
)

trees = train_on_binned(
  X_binned, y,
  weights                 = weights,
  iteration_callback      = make_callback(),
  iteration_count         = 1000000,
  min_data_weight_in_leaf = 2.0,
  learning_rate           = 0.3,
  bagging_temperature     = 0.0, # otherwise, non-deterministic
  mpi_comm                = comm,
);

iteration_count = length(trees) - 1 + max_iterations_without_improvement

print("Rank $rank, $iteration_count iterations (expected 673)\n")

MPI.Barrier(comm)
@assert iteration_count == 673 # this will segfault MPI if it fails. oh well
MPI.Barrier(comm)
