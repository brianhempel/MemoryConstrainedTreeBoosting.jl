def bench
  times = `make profile_hrrr`.scan(/[\d\.]+ seconds/).map(&:to_f)
  mean_time = times.sum / times.size
  puts "#{mean_time} seconds"
  mean_time
end

def bench_chunk_sizes(chunk_size_1, chunk_size_2)
  print "#{chunk_size_1}\t#{chunk_size_2}\t"
  code = File.read("src/MemoryConstrainedTreeBoosting.jl")
  code.sub!(/is_chunk_size = isa\(leaf_is, UnitRange\) \? \d+ : \d+/, "is_chunk_size = isa(leaf_is, UnitRange) ? #{chunk_size_1} : #{chunk_size_2}")
  File.write("src/MemoryConstrainedTreeBoosting.jl", code)
  sleep 15
  bench
end

def bench_feature_chunk_size(chunk_size)
  print "#{chunk_size}\t"
  code = File.read("src/MemoryConstrainedTreeBoosting.jl")
  code.sub!(/features_chunk_size = \d+/, "features_chunk_size = #{chunk_size}")
  File.write("src/MemoryConstrainedTreeBoosting.jl", code)
  sleep 15
  bench
end

class Integer
  def to_chunk_size
    (10000.0*(1.2**self) / 64).round * 64
  end
end

# binding.irb

# bench_chunk_sizes(8704, 448)
# bench_chunk_sizes(8704, 448)
# bench_chunk_sizes(8704, 448)

best_chunk_size_1 = (-20..20).map(&:to_chunk_size).min_by do |chunk_size|
  bench_chunk_sizes(chunk_size, 320)
end

puts "Best chunk size 1: #{best_chunk_size_1}"

best_chunk_size_2 = (-20..20).map(&:to_chunk_size).min_by do |chunk_size|
  bench_chunk_sizes(best_chunk_size_1, chunk_size)
end

puts "Best chunk size 2: #{best_chunk_size_2}"

bench_chunk_sizes(best_chunk_size_1, best_chunk_size_2)

best_feature_chunk_size = [4,6,8,10,12,16,24,32,48,64,96,128,192,256,384,512,768,1024].min_by do |chunk_size|
  bench_feature_chunk_size(chunk_size)
end

puts "Best feature chunk size: #{best_feature_chunk_size}"

