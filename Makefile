tests:
	JULIA_NUM_THREADS=${CORE_COUNT} julia --project=. test/Test.jl

profile:
	# May need to turn threads off in various functions
	# julia --project=. test/Profile.jl
	JULIA_NUM_THREADS=${CORE_COUNT} julia --project=. test/Profile.jl

profile_hrrr:
	# May need to turn threads off in various functions
	# julia --project=. test/Profile.jl
	JULIA_NUM_THREADS=${CORE_COUNT} julia --project=. test/ProfileHRRR.jl

evaluation:
	# May need to turn threads off in various functions
	julia --project=. test/Evaluation.jl
	# JULIA_NUM_THREADS=${CORE_COUNT} julia --project=. test/Evaluation.jl

distributed_test:
	# May need to turn threads off in various functions
	mpiexecjl -n 3 julia --project=. test/DistributedTest.jl
	# JULIA_NUM_THREADS=${CORE_COUNT} mpiexecjl -n 3 julia --project=. test/DistributedTest.jl

distributed_profile:
	# May need to turn threads off in various functions
	# mpiexecjl -n 1 julia --project=. test/DistributedProfile.jl
	JULIA_NUM_THREADS=${CORE_COUNT} mpiexecjl -n 3 julia --project=. test/DistributedProfile.jl

distributed_evaluation:
	# May need to turn threads off in various functions
	mpiexecjl -n 3 julia --project=. test/DistributedEvaluation.jl
	# JULIA_NUM_THREADS=${CORE_COUNT} mpiexecjl -n 2 julia --project=. test/DistributedEvaluation.jl
