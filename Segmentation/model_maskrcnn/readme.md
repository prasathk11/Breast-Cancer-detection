Training::
    
    from model.training import train
    train()

Evaluation::

    from model.inference import InferenceModel, DataModule
    m = InferenceModel("/path/to/weights")
    data = DataModule()
    ypred, ytrue = m.evaluate(data.test_dataloader())
    ypred = m.postprocess(ypred, minscore=.5)
    # rest of evaluation

Prediction::
    
    from model import InferenceModel
    m = InferenceModule("/path/to/weights")
    m.predict(image)

##################################################################################

Training
    https://ui.neptune.ai/simonm3/envision/e/EN-243/charts
    https://ui.neptune.ai/simonm3/envision/experiments?viewId=de550116-70fd-4893-b987-4a22a6238081

    _outputs = "https://drive.google.com/drive/folders/12xgGv9MOn5BK9S4KUvB4UTT_Ohbb8AJE?usp=sharing"
    weights = "envision_EN-243_epoch=8-val_loss=0.35.pt"
    evaluation = "maskrcnn_243_test/evaluate.ipynb" 
    

