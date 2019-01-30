import TranscodingStreams, CodecZstd, CodecLz4

one_gb = rand(UInt8, 2*1024*1024*1024)

function compress(bytes)
  # TranscodingStreams.transcode(CodecZstd.ZstdCompressor, bytes)
  # TranscodingStreams.transcode(CodecLz4.LZ4Compressor, bytes)

  # Annoying that you have to do it this way. (c.f. https://bicycle1885.github.io/TranscodingStreams.jl/latest/examples.html#Explicitly-finish-transcoding-by-writing-TOKEN_END-1)
  buffer = IOBuffer()
  stream = CodecZstd.ZstdCompressorStream(buffer)
  write(stream, bytes, TranscodingStreams.TOKEN_END)
  flush(stream)
  compressed = take!(buffer)
  close(stream)

  compressed
end

function decompress(bytes)
  TranscodingStreams.transcode(CodecZstd.ZstdDecompressor, bytes)
  # TranscodingStreams.transcode(CodecLz4.LZ4Decompressor, bytes)
end

compressed = compress(one_gb)
println("Compressing...")
@time compressed = compress(one_gb)

decompressed = decompress(compressed)
println("Decompressing...")
@time decompressed = decompress(compressed)

println("Successful?")
println(decompressed == one_gb)
