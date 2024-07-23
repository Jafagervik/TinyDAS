using FilePathsBase
using Base.Threads

function rename_files_in_directory_parallel(directory::AbstractString, prefix::AbstractString)
    files = readdir(directory)
    @threads for file in files
        !startswith(file, prefix) && continue

        old_path = joinpath(directory, file)
        new_name = file[length(prefix)+1:end]
        new_path = joinpath(directory, new_name)
        try
            mv(old_path, new_path)
            println("Renamed: $old_path to $new_path")
        catch e
            println("Error renaming $old_path: $e")
        end
    end
end

# Define the directory and prefix
directory = "C:\\Users\\jaf\\Documents\\DAS\\infer"  # Replace with your directory path
prefix = "FORESEE_UTC_"

# Rename the files
rename_files_in_directory_parallel(directory, prefix)