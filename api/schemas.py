from pydantic import BaseModel
        
class TrackerRequest(BaseModel):
    input_folder: str 
    output_folder: str
    video_name: str = "VID_20230322_173233"
    draw_tracking: bool = False
    draw_circles: bool = False
    radius: float = 10