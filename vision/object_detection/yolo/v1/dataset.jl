using JLD2
using Images
import DataFrames
using CSV

function readdataset(csvfile, imgdir, labeldir; S=7, B=2, C=20, idx=Colon())
    annotations = CSV.read(csvfile, DataFrames.DataFrame; header=false)


    # read images
    imgs = []
    for imgname in annotations.Column1[idx]
        img = Images.load(joinpath(imgdir, imgname))

        # resize the images
        img = Images.imresize(img, (448, 448))
        img = channelview(img)
        img = permutedims(img, (3,2,1))

        push!(imgs, Float32.(img))
    end
    imgs = cat(imgs..., dims=4)

    # read label
    label_matrices = []
    for labelname in annotations.Column2[idx]
        labels = readlines(joinpath(labeldir, labelname))
        # class_number, x, y, w, h
        boxes = map(labels) do lines
            map(enumerate(split(lines, r"\s+"))) do (idx, v)
                if idx == 1
                    parse(Int, v)
                else
                    parse(Float32, v)
                end
            end
        end
        
        # handle the label
        label_matrix = zeros(Float32, S, S, 5B+C)
        for box in boxes
            class, x, y, w, h = box
            i, j = (Int∘floor)(S * y) + 1, (Int∘floor)(S * x) + 1
            x_cell, y_cell = x*S - (j-1), S*y - (i-1)
            width_cell, height_cell = w*S, h*S

            if label_matrix[i, j, 21] == 0  # if two bounding boxes corrupt
                label_matrix[i, j, 21] = 1
                label_matrix[i, j, 22:25] .= [x_cell, y_cell, width_cell, height_cell] 
                label_matrix[i, j, class] = 1
            end
        end

        push!(label_matrices, label_matrix)
    end
    label_matrices = cat(label_matrices..., dims=4)

    return imgs, label_matrices
end


