include("utils.jl")

using Flux
using Flux: @functor, onecold
using Flux.Losses: mse

struct YOLOv1
    darknet
    mlp

    darknet_pretrained
end

Flux.params(yolo::YOLOv1) = if yolo.darknet_pretrained
    Flux.params(yolo.mlp)
else
    Flux.params(yolo.darknet, yolo.mlp)
end


function cnnblock(ker_size, channel_pair, pad=0, stride=1)
    Chain(
        Conv((ker_size, ker_size), channel_pair; stride=stride, pad=pad),
        BatchNorm(channel_pair[2], x -> leakyrelu(x, 0.1)),
    )
end

function YOLOv1()
    darknet = Chain(
        # (448, 448, 3)
        cnnblock(7, 3 => 64, 3, 2),
        MaxPool((2, 2)),
        cnnblock(3, 64 => 192, 1),
        MaxPool((2, 2)),
        cnnblock(1, 192 => 128),
        cnnblock(3, 128 => 256, 1),
        cnnblock(1, 256 => 256),
        cnnblock(3, 256 => 512, 1),
        MaxPool((2, 2)),
        [Chain(
            cnnblock(1, 512 => 256),
            cnnblock(3, 256 => 512, 1),
        ) for _ in 1:4]...,
        cnnblock(1, 512 => 512),
        cnnblock(3, 512 => 1024, 1),
        MaxPool((2, 2)),
        [Chain(cnnblock(1, 1024 => 512), cnnblock(3, 512 => 1024, 1)) for _ in 1:2]...,
        cnnblock(3, 1024 => 1024, 1),
        cnnblock(3, 1024 => 1024, 1, 2),
        cnnblock(3, 1024 => 1024, 1),
        cnnblock(3, 1024 => 1024, 1),
    )

    mlp = Chain(
        Dense(7*7*1024, 496, x -> leakyrelu(x, 0.1)),
        Dropout(0.0),
        Dense(496, 7*7*30),
        x -> reshape(x, 7, 7, 30, :)
    )

    YOLOv1(darknet, mlp, false)
end

function FastYOLOv1()
    darknet = Chain(
        # same parameters, but only 9 convolutional layers
        cnnblock(7, 3 => 64, 3, 2),
        MaxPool((2, 2)),
        cnnblock(3, 64 => 192, 1),
        MaxPool((2, 2)),
        cnnblock(3, 192 => 256, 1),
        cnnblock(3, 256 => 512, 1),
        MaxPool((2, 2)),
        cnnblock(3, 512 => 512, 1),
        cnnblock(3, 512 => 1024, 1),
        MaxPool((2, 2)),
        cnnblock(3, 1024 => 1024, 1),
        cnnblock(3, 1024 => 1024, 1, 2),
        cnnblock(3, 1024 => 1024, 1),
    )
    mlp = Chain(
        Dense(7*7*1024, 496, x -> leakyrelu(x, 0.1)),
        Dropout(0.0),
        Dense(496, 7*7*30),
        x -> reshape(x, 7, 7, 30, :)
    )

    YOLOv1(darknet, mlp, false)
end

function PretrainedYOLOv1(pretrained_darknet, mid_dim)
    mlp = Chain(
        Dense(mid_dim, 496, x -> leakyrelu(x, 0.1)),
        Dropout(0.0),
        Dense(496, 7*7*30),
        x -> reshape(x, 7, 7, 30, :)       
    )
    YOLOv1(pretrained_darknet, mlp, true)
end

function (m::YOLOv1)(x)
    m.mlp(Flux.flatten(m.darknet(x)))
end

"""
    compute the loss between the bounding boxes and ground truth

For ŷ, it has shape of 7 x 7 x 30 x batchsize, where 30 = 20(for class probabilities) + 2*5(object confidence, x, y, w, h)
For y, its shape is    7 x 7 x 25 x batchsize, where 25 = 20(for class probabilities) + 5(object existency, x, y, w, h)
"""
function loss_yolov1(ŷ, y; λcoord=5f0, λnoobj=0.5f0, ε=1f-6)
    
    box1 = ŷ[:, :, 22:25, :]
    box2 = ŷ[:, :, 27:30, :]
    box_label = y[:, :, 22:25, :]

    # get 7 x 7 x batchsize iou_box
    iou_b1 = intersection_over_union(box1, box_label)
    iou_b2 = intersection_over_union(box2, box_label)

    iou_b1_v = reshape(iou_b1, 1, :)
    iou_b2_v = reshape(iou_b2, 1, :)

    # get 49*2, batchsize ious
    ious = vcat(iou_b1_v, iou_b2_v)
    bestbox = reshape(onecold(ious, [0f0, 1f0]), 7, 7, 1, :)

    exists_box = y[:, :, 21:21, :]

    # box coordinates, shape of 7 x 7 x 4 x batchsize
    box_predictions = @. (box2 * bestbox + box1 * (1 - bestbox)) * exists_box
    box_targets = exists_box .* box_label
    
    xy_pred = box_predictions[:, :, 1:2, :]  # this may be negative, but no need to process
    wh_pred = box_predictions[:, :, 3:4, :]  # this may be negative, and may be zero, need get its abs value for sqrt
                                             # also a small number, since dx^{1/2} = 1/(2x^{1/2})dx
    xy_label = box_targets[:, :, 1:2, :]
    wh_label = box_targets[:, :, 3:4, :]


    coord_loss = mse(xy_pred, xy_label) + mse(sqrt.(abs.(wh_pred) .+ ε), sqrt.(wh_label))

    # confidence loss
    confidence_predictions = @. (ŷ[:, :, 26:26, :] * bestbox + ŷ[:, :, 21:21, :] * (1 - bestbox)) * exists_box
    confidence_predictions_noobj = @. (ŷ[:, :, 21:21, :] + ŷ[:, :, 26:26, :]) * (1 - exists_box)
    confidence_targets = exists_box .* y[:, :, 21:21, :]
    confidence_targets_noobj = (1f0 .- exists_box) .* y[:, :, 21:21, :]
    obj_loss = mse(confidence_predictions, confidence_targets) + 
       λnoobj * mse(confidence_predictions_noobj, confidence_targets_noobj)

    # class loss
    class_loss = mse(
        exists_box .* ŷ[:, :, 1:20, :],
        exists_box .* y[:, :, 1:20, :]
    )

    # sum the losses
    λcoord * coord_loss + obj_loss + class_loss 
end
