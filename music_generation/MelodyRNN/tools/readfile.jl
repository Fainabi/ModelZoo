abstract type AbstractFileReader end

# this filereader can read files iteratively
# using BFS
mutable struct FileReader <: AbstractFileReader
    basepath::AbstractString
    filetype::AbstractString
    files::Vector
    dirs::Vector

    function FileReader(path::AbstractString; filetype=".mid")
        @assert (filetype == ".mid") "Not implement generic file reader"

        files = []
        dirs = []
        push!(dirs, path)

        new(path, filetype, files, dirs)
    end
end


function Base.iterate(fr::FileReader)
    # file buffer is not empty, read and return that file
    if !isempty(fr.files)
        filename = pop!(fr.files)

        # default using MIDI.load function
        try
            fileloaded = load(filename)
            return fileloaded
        catch
            @warn "omit midi file cannot read"
            return iterate(fr)
        end
    end

    # directory buffer is not empty
    if !isempty(fr.dirs)
        dir_name = pop!(fr.dirs)

        for name in readdir(dir_name, join=true)

            # tell if is a file or a directory
            if isfile(name) && endswith(name, fr.filetype)
                push!(fr.files, name)
            elseif isdir(name)
                push!(fr.dirs, name)
            end
        end

        return iterate(fr)
    end

    nothing
end

function Base.show(io::IO, fr::FileReader)
    if isempty(fr.filetype)
        print("FileReader with any file type")
    else
        print(io, "FileReader with extension ", fr.filetype)
    end
end

function reset(fr::FileReader)
    empty!(fr.files)
    empty!(fr.dirs)
    push!(fr.dirs, fr.basepath)
end

function Base.collect(fr::FileReader, readnum::Integer)
    res = []
    readed_num = 0

    while readed_num < readnum
        f = iterate(fr)

        if isnothing(f)
            break
        end

        push!(res, f)
        readed_num += 1
    end

    res
end

function Base.collect(fr::FileReader)
    res = []

    while true
        f = iterate(fr)

        if isnothing(f)
            break
        end

        push!(res, f)
    end

    res
end

