# to onnx
python inference/ckpt_to_onnx.py 

# launch web service
uvicorn service.service:app