class DefaultConfig():
    target_emotion_map = {'anger': 0,
                          'fear': 1,
                          'happy': 2,
                          'sadness': 3,
                          'surprise': 4,
                          'neutral': 5}
    target_image_size = (64, 64)
    out_channels = 1
    batch_size = 64
    input_shape = (64, 64, 3)
    l2_regularization = 0.0001
    learning_rate = 0.001
    epochs = 1000
    num_classes = len(target_emotion_map.keys())
    validation_split = 0.1
    log_file_path = '/home/user/Documents/delta/expression/KerasFaceExpression/log'
    model_path = '/home/user/Documents/delta/expression/KerasFaceExpression/model'
    data_path = '/home/user/Documents/dataset/NovaEmotions'
    dataset_name = 'NovaEmotions'
    model_name = 'Mini_Xception'
