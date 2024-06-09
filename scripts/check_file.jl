using HDF5

fpath = "..\\..\\DAS\\Test\\20200301_000015.hdf5"

h5open(fpath, "r") do f
    d = f["raw"]
    ts = f["timestamp"]
    @show size(d)
    @show length(ts)
end