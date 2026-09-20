import base64, json, re
import cv2
from langchain_nvidia_ai_endpoints import ChatNVIDIA


client = ChatNVIDIA(
  model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
  api_key="$NVIDIA_API_KEY",
  temperature=0.6,

  top_p=0.95,

  max_completion_tokens=65536,
)

lc_messages = [
  {
    "role": "user",
    "content": [
      {
        "type": "text",
        "text": "What is in this image?",
      },
      {
        "type": "image_url",
        "image_url": {
          "url": "https://assets.ngc.nvidia.com/products/api-catalog/phi-3-5-vision/example1b.jpg",
        },
      },
    ],
  },
]

response = client.invoke(lc_messages)
if response.additional_kwargs and "reasoning_content" in response.additional_kwargs:
  print(response.additional_kwargs["reasoning_content"])
print(response.content)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 40)

def grab():
    for _ in range(3):              
        cap.grab()
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("camera read failed")
    return frame

def to_data_url(frame):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

def locate_object(frame, target):
    h, w = frame.shape[:2]
    prompt = (f"Locate the object '{target}' in an image of size {w}x{h}"
             'Reply with JSON only: {"found": true/false, "x": int, "y": int, '
              '"bbox": [x1, y1, x2, y2]} where x,y is the object center in pixel coordinates.')

    r = client.chat.completions.create(
        model=client.model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": to_data_url(frame)}},
        ]}],
        max_tokens=256, temperature=0.2,
        extra_body={"top_k": 1, "chat_template_kwargs": {"enable_thinking": False}},
    )
    m = re.search(r"\{.*\}", r.choices[0].message.content, re.S)

    if m:
        return json.loads(m.group(0))
    
    return {"found": False}
