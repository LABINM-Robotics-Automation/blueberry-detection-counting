def detect_on_image(
    input_image,
    conf_threshold,
    model
):
    '''
    This function receives an image and object detection model and returns
    input image with the detected objects

    Args:
        input_image    : input image loaded with opencv
        conf_threshold : float between 0 and 1
        model          : object detection model that will be used to detect
    '''

    prediction = model.detect(input_image, conf_threshold)

    image_with_objects_detected = model.plot_prediction(input_image, prediction)

    return image_with_objects_detected


