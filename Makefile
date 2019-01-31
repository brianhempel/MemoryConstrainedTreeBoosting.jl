run_tests:
	JULIA_NUM_THREADS=4 julia --project=. test/Test.jl

compression_test:
	JULIA_NUM_THREADS=4 julia --project=. test/CompressionTest.jl