from pydantic import BaseModel


class ResearchPlan(BaseModel):
    objective: str
    subquestions: list[str]
    relevant_cities: list[str]
