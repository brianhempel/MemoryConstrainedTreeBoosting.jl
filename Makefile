run_tests:
	JULIA_NUM_THREADS=4 julia --project=. test/Test.jl

compression_test:
	JULIA_NUM_THREADS=4 julia --project=. test/CompressionTest.jl

profile:
	# May need to turn threads off in apply_trees and perhaps_split_tree
	julia --project=. test/Profile.jl
	# JULIA_NUM_THREADS=4 julia --project=. test/Profile.jl
