using HDF5
using Dates
using Distributed

addprocs(16)

@everywhere using Dates
@everywhere using HDF5

@everywhere function split_hdf5_file(file_path::String, segment_duration::Int, total_duration::Int)
    num_segments = total_duration ÷ segment_duration
    original_filename = first(splitext(basename(file_path)))
    date_str, time_str = split(original_filename, '_')
    segment_length = 625

    initial_timestamp = DateTime(date_str * time_str, "yyyymmddHHMMSS")

    raw_data, timestamp_data = h5open(file_path, "r") do f
        read(f["raw"]), read(f["timestamp"])
    end

    rm(file_path)

    for i in 0:(num_segments-1)
        bf = i * segment_length+1
        n = (i+1)*segment_length

        raw_segment = raw_data[bf:n, :]
        timestamp_segment = timestamp_data[bf:n]

        new_timestamp = initial_timestamp + Second(i * segment_duration)
        new_filename = string(Dates.format(new_timestamp, "yyyymmdd_HHMMSS"), ".hdf5")
        new_file_path = joinpath(dirname(file_path), new_filename)

        h5write(new_file_path, "raw", raw_segment)
        h5write(new_file_path, "timestamp", timestamp_segment)
    end

    println("Done with $original_filename")
end

function process_directory(directory::String, segment_duration::Int, total_duration::Int)
    files = filter(x -> endswith(x, ".hdf5"), readdir(directory, join=true))
    @sync for file in files
        @spawn split_hdf5_file(file, segment_duration, total_duration)
    end
    "DONE" |> println
end

directory = "..\\..\\DAS\\2023"
segment_duration = 5 
total_duration = 600 

process_directory(directory, segment_duration, total_duration)
