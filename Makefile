run_tests:
	JULIA_NUM_THREADS=$CORE_COUNT julia --project=. test/Test.jl

profile:
	# May need to turn threads off in various functions
	# julia --project=. test/Profile.jl
	JULIA_NUM_THREADS=$CORE_COUNT julia --project=. test/Profile.jl

profile_hrrr:
	# May need to turn threads off in various functions
	# julia --project=. test/Profile.jl
	JULIA_NUM_THREADS=$CORE_COUNT julia --project=. test/ProfileHRRR.jl

evaluate:
	# May need to turn threads off in various functions
	julia --project=. test/Evaluation.jl
	# JULIA_NUM_THREADS=$CORE_COUNT julia --project=. test/Evaluation.jl
