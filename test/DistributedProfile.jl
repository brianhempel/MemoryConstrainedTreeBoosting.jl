
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


# For SREF, datapoint/feature ratio is ~370:1
# For HRRR, 530:1
feature_count = 4000
point_count   = div(feature_count, rank_count)*400 # don't explode my machine

Random.seed!(123456)

X_binned_full = rand(UInt8(1):UInt8(255), (point_count, feature_count))
y_full        = round.(rand(MemoryConstrainedTreeBoosting.Prediction, point_count))
weights_full  = rand(MemoryConstrainedTreeBoosting.DataWeight, point_count)

Random.seed!(123456 + rank)

# Simulate data being on different nodes
my_chunk_range = chunk_range(rank+1, rank_count, length(y_full))
X_binned       = X_binned_full[my_chunk_range, :]
y              = y_full[my_chunk_range]
weights        = weights_full[my_chunk_range]

validation_range    = 1:div(point_count,3)
my_validation_range = chunk_range(rank+1, rank_count, length(validation_range))
validation_X_binned = (@view X_binned_full[validation_range, :])[my_validation_range, :]
validation_y        = (@view y_full[validation_range])[my_validation_range]
validation_weights  = (@view weights_full[validation_range])[my_validation_range]

function make_iteration_callback()
  make_callback_to_track_validation_loss(
    validation_X_binned,
    validation_y;
    validation_weights = validation_weights
  )
end

duration = @elapsed train_on_binned(X_binned, y; mpi_comm = comm, weights = weights, iteration_count = 10, feature_fraction = 0.9, min_data_weight_in_leaf = 30000, max_leaves = 6, bagging_temperature = 0.5, iteration_callback = make_iteration_callback())

MPI.Barrier(comm)
if rank == root
  println()
  println("\n$duration seconds.")
  println()
end

