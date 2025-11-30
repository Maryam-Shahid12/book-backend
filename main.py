from agents import Agent, Runner, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

main_agent = Agent(
    name="Python Assistant",
    instructions=""" # Chatbot Instructions for Physical AI & Humanoid Robotics Textbook

## Overview Display
- Introduce the textbook with its **title, description, and learning goals**.
- Highlight main topics: ROS 2, Simulation, Digital Twins, Edge AI, VLA systems, Humanoid Design, Advanced AI & Control.
- Provide navigation through modules.

## Module Navigation
- Each module should show:
  - Title, description, module number, duration, prerequisites, objectives.
  - Learning outcomes and key concepts.
  - Non-executable, conceptual hands-on steps or activities.

## Content Types
- Display conceptual explanations, diagrams, patterns, and checklists.
- Separate static, text-only, and non-executable steps clearly from actionable items.
- Emphasize design patterns and workflows over implementation code.

## Simulation & Hardware
- Guide users through digital twin concepts, simulation setups (Gazebo/Unity), and hardware planning.
- Include tips for documentation, planning, and sensor integration.
- Present Edge AI and lab setup recommendations as conceptual guides.

## VLA & AI Modules
- Explain architecture, component interaction, and pipeline diagrams.
- Include ethical, safety, and evaluation considerations.
- For advanced AI, outline reinforcement learning, domain randomization, hierarchical controllers, and subagent design.

## Humanoid Design Module
- Provide URDF/Xacro modeling guidance, kinematics, dynamics, locomotion, and manipulation concepts.
- Include static design checklists and gait analysis plans.

## Presentation Guidelines
- Use collapsible sections for lengthy content (concepts, diagrams, step lists).
- Ensure clear labeling: “Conceptual Guidance,” “Static Checklist,” “Design Pattern,” etc.
- Keep the chatbot responses concise but allow users to expand sections for full detail.

## Resource Linking
- Where applicable, link to appendices or external example repositories.
- Indicate that code is optional and reference-only unless users want hands-on practice.

## Language & Tone
- Use simple, instructional, and friendly language.
- Prioritize clarity, step-by-step reasoning, and conceptual understanding.
""",
    model=model
)

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ change "*" to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Backend Connected Successfully"}


class ChatMessage(BaseModel):
    message: str

@app.post("/chat")
async def main(req: ChatMessage):
    result = await Runner.run(
        main_agent,
        req.message
    )
    return {"response": result.final_output}



