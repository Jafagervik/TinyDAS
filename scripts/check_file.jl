using HDF5

function load_das_file(filename::String)
    h5open(filename, "r") do f
        return size(f["raw"][])
    end
end

struct Dataset
    path::String
    transpose::Bool
    data::Nothing

    function Dataset(;path::String = "./data", transpose::Bool = false, n::Int = -1, start::Int = 20000)
        self = new(path, transpose, nothing)
        self.data = skips(self, start)
        return self
    end
end

function run(f::String, c::Int, i::Int, start::Int)::Int
    s = load_das_file(f)
    if s != (625, 2137)
        @show s, f 
        c += 1
    end
    i % 2000 == 0 && @show (i + start, c)
    #i += 1
    return c
end

function skips(self::Dataset, start::Int)::Int
    c = 0
    #i = 0
    filenames = readdir(self.path, join=true)
    for (i,f) in enumerate(filenames[start:end])
        c += @inbounds run(f, i, c, start)
    end

    println("Done")
    return c
end

ds = Dataset(;start=20000)
println(ds.data)