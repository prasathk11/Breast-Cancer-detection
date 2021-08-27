import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

class AnomalyModel(object):
    "MobileNetV2"
    def __init__(self,model_path):
        self.idx2class = ['anomaly','normal']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = torch.load(model_path,map_location=self.device)
        self.model  = self.model.eval().to(self.device)

        self.transform= T.Compose([
                        T.ToTensor(),
                        T.Resize((224, 224)),
                        T.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])])

    @torch.no_grad()
    def __call__(self,image):
        """
        input: ndarray uint8 (H,W,3)
        output: dict{label:str,score:float}
        """
        #Predict
        out = self.transform(image).unsqueeze(0).to(self.device)    
        out = self.model(out)   
        out = F.softmax(out,dim=-1)

        pred_class = self.idx2class[torch.argmax(out).item()]
        pred_proba = torch.max(out).item()
                
        return {"label":pred_class,
                "score:":pred_proba}
