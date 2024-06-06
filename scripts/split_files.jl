using HDF5
using Dates
using Base.Threads

function split_hdf5_file(file_path::String, segment_duration::Int, total_duration::Int)
    # Constants
    num_segments = total_duration ÷ segment_duration
    original_filename = splitext(basename(file_path))[1]
    date_str, time_str = split(original_filename, '_')
    segment_length = 2500

    # Parse the initial timestamp
    initial_timestamp = DateTime(date_str * time_str, "yyyymmddHHMMSS")

    # Open the original HDF5 file and read the content
    raw_data, timestamp_data = h5open(file_path, "r") do f
        read(f["raw"]), read(f["timestamp"])
    end

    # Delete the original file
    rm(file_path)

    # Split and save the data into new files
    for i in 0:(num_segments-1)
        raw_segment = raw_data[i*segment_length+1:(i+1)*segment_length, :]
        timestamp_segment = timestamp_data[i*segment_length+1:(i+1)*segment_length]

        # Calculate new timestamp for the segment
        new_timestamp = initial_timestamp + Second(i * segment_duration)
        new_filename = string(Dates.format(new_timestamp, "yyyymmdd_HHMMSS"), ".hdf5")
        new_file_path = joinpath(dirname(file_path), new_filename)

        h5write(new_file_path, "raw", raw_segment)
        h5write(new_file_path, "timestamp", timestamp_segment)
        println("Created: $new_file_path")
    end
end

function process_directory(directory::String, segment_duration::Int, total_duration::Int)
    files = filter(x -> endswith(x, ".hdf5"), readdir(directory, join=true))
    @threads for file in files
        split_hdf5_file(file, segment_duration, total_duration)
    end
end

# Define the directory, segment duration, and total duration
directory = "../../DAS/2023"  # Replace with your directory path
segment_duration = 20  # Duration of each segment in Seconds
total_duration = 600  # Total duration of the original file in seconds

# Start the processing
process_directory(directory, segment_duration, total_duration)