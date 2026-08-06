from pydantic import BaseModel, Field

class Student(BaseModel):
    name: str = "Harshith"
    age: int = 20
    grade: str = "A"
    cgpa: float = Field(
        ...,
        gt=0,
        lt=10,
        description="CGPA must be between 0 and 10"
    )

new_student = {
    "age": 20,
    "grade": "A",
    "cgpa": 8.7
}

student = Student(**new_student)

student_json = student.model_dump_json(indent=4)
print(student_json)

print(student)