import torch
from models.semr_physics_pro import SEMRPhysicsPro

def export():
    model = SEMRPhysicsPro()
    model.load_state_dict(torch.load('pretrained/best.pth'))
    model.eval()
    dummy = torch.randn(1, 1, 256, 256)
    torch.onnx.export(model, dummy, "semr_physics.onnx",
                      input_names=['input'], output_names=['output','uncertainty'],
                      opset_version=14, dynamic_axes={'input':{2:'h',3:'w'}})
    print("ONNX exported.")
if __name__ == '__main__':
    export()