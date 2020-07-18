run_tests:
	JULIA_NUM_THREADS=4 julia --project=. test/Test.jl

test_regression:
	JULIA_NUM_THREADS=4 julia --project=. test/RegressionTest.jl

compression_test:
	JULIA_NUM_THREADS=4 julia --project=. test/CompressionTest.jl

profile:
	# May need to turn threads off in various functions
	# julia --project=. test/Profile.jl
	JULIA_NUM_THREADS=4 julia --project=. test/Profile.jl

evaluate:
	# May need to turn threads off in various functions
	# julia --project=. test/Evaluation.jl
	JULIA_NUM_THREADS=4 julia --project=. test/Evaluation.jl
