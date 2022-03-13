using Images
using ImageView
using ImageDraw
using Flux: unsqueeze, cpu

"""
    intersection_over_union(boxes_pred, boxes_labels)

Compute the intersection area over union area bwteen the two boxes. 
"""
function intersection_over_union(boxes_pred, boxes_labels)
    # input 7 x 7 x 4 x batchsize
    #   the five scalars are: x, y, w, h 

    area_predicted = boxes_pred[:, :, 3, :] .* boxes_pred[:, :, 4, :]
    area_labels = boxes_labels[:, :, 3, :] .* boxes_labels[:, :, 4, :]

    # compute the common area
    box1_x1 = boxes_pred[:, :, 1, :] .- boxes_pred[:, :, 3, :] ./ 2
    box1_x2 = boxes_pred[:, :, 1, :] .+ boxes_pred[:, :, 3, :] ./ 2
    box1_y1 = boxes_pred[:, :, 2, :] .- boxes_pred[:, :, 4, :] ./ 2
    box1_y2 = boxes_pred[:, :, 2, :] .+ boxes_pred[:, :, 4, :] ./ 2

    box2_x1 = boxes_labels[:, :, 1, :] .- boxes_labels[:, :, 3, :] ./ 2
    box2_x2 = boxes_labels[:, :, 1, :] .+ boxes_labels[:, :, 3, :] ./ 2
    box2_y1 = boxes_labels[:, :, 2, :] .- boxes_labels[:, :, 4, :] ./ 2
    box2_y2 = boxes_labels[:, :, 2, :] .+ boxes_labels[:, :, 4, :] ./ 2

    x1 = max.(box1_x1, box2_x1)
    y1 = max.(box1_y1, box2_y1)
    x2 = min.(box1_x2, box2_x2)
    y2 = min.(box1_y2, box2_y2)

    intersection = @. max(0f0, x2 - x1) * max(0f0, y2 - y1)

    @. intersection / (area_labels + area_predicted - intersection + 1f-6)
end


"""
    mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5; num_classes=20)

Compute the accuracy of the predicted boxes matching the true boxes.
"""
function mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5; num_classes=20)
    
    for c in 1:num_classes
        detections = []
        ground_truths = []
    end
end

"""
    non_max_suppression(bboxes, iou_threshold, threshold)

Filter out the bboxes with lower confidence under the threshold. No batches.
"""
function non_max_suppression(bboxes, iou_threshold=0.5, threshold=0.5)
    # confidence should be above threshold
    bboxes = filter(bboxes) do tp
        tp[2] > threshold
    end

    # iou confidence
    bboxes = sort(bboxes, by=x -> x[2])  # sort by confidence
    bboxes_after_nms = []

    while !isempty(bboxes)
        chosen_box = pop!(bboxes)
        push!(bboxes_after_nms, chosen_box)

        filter!(bboxes) do box
            box[1] != chosen_box[1] || begin  # different class
                iou = intersection_over_union(
                    reshape(chosen_box[3:end], 1, 1, :, 1), 
                    reshape(box[3:end], 1, 1, :, 1))
                iou[1]
            end < iou_threshold  # different object
        end
    end

    bboxes_after_nms
end


"""
    get_bboxes(out_mat)

The output matrix is from yolo model, and here decode it to get the bounding boxes.
"""
function get_bboxes(predictions; S=7)
    predictions = cpu(predictions)

    bboxes1 = predictions[:, :, 22:25, :]
    bboxes2 = predictions[:, :, 27:30, :]
    scores = permutedims(predictions[:, :, [21,26], :], (1,2,4,3))
    scores = permutedims(flatten(scores), (2,1))
    
    best_box = reshape(onecold(scores, 0:1), S, S, 1, :)

    # concatenate the boxes best
    cell_indices = unsqueeze(repeat(collect(0:S-1), 1, 7, size(predictions)[4]), 3)
    best_boxes = @. bboxes1 * (1 - best_box) + bboxes2 * best_box
    wh = best_boxes[:, :, 3:4, :] ./ S
    y = (best_boxes[:, :, 2, :] .+ cell_indices) ./ S
    x = (best_boxes[:, :, 1, :] .+ permutedims(cell_indices, (2,1,3,4))) ./ S

    predicted_class = permutedims(predictions[:, :, 1:20, :], (1,2,4,3))
    predicted_class = permutedims(flatten(predicted_class), (2, 1))
    predicted_class = reshape(onecold(predicted_class, 1:20), S, S, 1, :)

    best_confidence = maximum(predictions[:, :, [21,26], :], dims=3)

    bboxes = cat(predicted_class, best_confidence, x, y, wh, dims=3)
    bboxes = map(1:size(bboxes)[end]) do idx
        boxes = reshape(bboxes[:, :, :, idx], S*S, :)
        boxes = map(1:S*S) do box_i
            boxes[box_i, :]
        end
    end
end

function plot_images(img, boxes; label=false)
    img = copy(img)
    hlen, wlen = size(img)

    for box in boxes
        box = box[3:end]  # omit the label

        x, y, w, h = box
        # should we remove the negative values boxes?
        
        ul_x = (Int∘floor)((x - w/2) * wlen + 1)
        ul_y = (Int∘floor)((y - h/2) * hlen + 1)
        lr_x = (Int∘floor)((x + w/2) * wlen + 1)
        lr_y = (Int∘floor)((y + h/2) * hlen + 1)

        # draw the rectangle
        rect = [(ul_x, ul_y), (ul_x, lr_y), (lr_x, lr_y), (lr_x, ul_y)]
        ImageDraw.draw!(img, Polygon(rect), RGB{N0f8}(0, 0, 1.0))
    end

    imshow(img)
end

function apply_yolo(model, img)
    img_reshaped = Images.imresize(img, 448, 448)
    img_channels = Float32.(permutedims(channelview(img_reshaped), (3,2,1)))
    img_channels = unsqueeze(img_channels, 4)

    out_mat = model(img_channels)
    boxes = get_bboxes(out_mat)
    bboxes_after_nms = non_max_suppression(boxes[1])

    plot_images(img, bboxes_after_nms)
end
