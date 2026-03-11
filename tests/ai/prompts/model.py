
from pydantic import BaseModel

# There is a problem with isinstance() during testing,
# classes have to be imported everywhere by full path
class InfoPageCreate(BaseModel):
    language: str
    level: str
    order: int
    type: str
    title: str
    content: str
