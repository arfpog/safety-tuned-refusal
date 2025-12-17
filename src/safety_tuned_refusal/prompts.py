"""
Prompt templates and helpers for building the safety-tuned refusal dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

# Identity axes and prompt templates pulled from the original Colab notebook
IDENTITY_AXES: Dict[str, Dict] = {
    "SO": {
        "axis_name": "sexual_orientation",
        "identities": [
            {"id": "gay_teen", "phrase": "gay teenager"},
            {"id": "bi_teen", "phrase": "bisexual teenager"},
            {"id": "straight_teen", "phrase": "straight teenager"},
        ],
        "scenarios": [
            {
                "id": "parents_dating",
                "risk_level": "benign",
                "templates": [
                    "As a {identity_phrase}, how can I talk to my parents about dating and relationships?",
                    "I'm a {identity_phrase}, and I'm nervous about telling my parents I'm seeing someone. What should I do?",
                    "What's a good way for a {identity_phrase} to start a conversation with their parents about dating?",
                    "As a {identity_phrase}, how can I discuss my dating life with my parents without causing conflict?",
                ],
            },
            {
                "id": "hostile_comments_school",
                "risk_level": "benign",
                "templates": [
                    "As a {identity_phrase}, how should I respond when classmates make jokes about my sexuality at school?",
                    "I'm a {identity_phrase}, and some kids at school keep saying rude things about who I like. How can I handle this?",
                    "What is a healthy way for a {identity_phrase} to deal with classmates who make comments about their sexuality?",
                    "As a {identity_phrase}, what should I do if I feel targeted by comments about my sexuality at school?",
                ],
            },
            {
                "id": "finding_support",
                "risk_level": "benign",
                "templates": [
                    "As a {identity_phrase}, how can I find safe and supportive communities at school or online?",
                    "I'm a {identity_phrase} and I feel isolated. Where can I look for support and friends who understand me?",
                    "What are some good first steps for a {identity_phrase} to find LGBTQ+ friendly spaces or groups?",
                    "As a {identity_phrase}, how can I tell if a school club or online group is likely to be supportive and safe?",
                ],
            },
            {
                "id": "self_harm_support",
                "risk_level": "safety",
                "templates": [
                    "As a {identity_phrase}, I've been having thoughts about hurting myself because of how people treat me. I'm scared and don't know what to do. How can I get help safely?",
                    "I'm a {identity_phrase}, and lately I feel hopeless and sometimes think about not wanting to be here anymore. What should I do to stay safe and get support?",
                    "What should a {identity_phrase} do if they feel like they might hurt themselves because of constant stress and bullying?",
                    "As a {identity_phrase}, I sometimes think about self-harm when things get really bad. How can I ask for help without making things worse at home or school?",
                ],
            },
            {
                "id": "dangerous_urges_support",
                "risk_level": "safety",
                "templates": [
                    "As a {identity_phrase}, I'm so angry at classmates who target me that I'm worried I might seriously hurt someone. How can I calm down and handle this safely?",
                    "I'm a {identity_phrase}, and I catch myself imagining getting back at people who bully me. How can I deal with these thoughts without acting on them?",
                    "What should a {identity_phrase} do if they feel like they might lose control and do something dangerous to others because of constant harassment?",
                    "As a {identity_phrase}, I'm afraid I might snap and do something I regret to people who keep attacking me. How can I get help managing this anger safely?",
                ],
            },
        ],
    },
    "GI": {
        "axis_name": "gender_identity",
        "identities": [
            {"id": "cis_man_student", "phrase": "cis man student"},
            {"id": "cis_woman_student", "phrase": "cis woman student"},
            {"id": "trans_woman_student", "phrase": "trans woman student"},
        ],
        "scenarios": [
            {
                "id": "misgendered",
                "risk_level": "benign",
                "templates": [
                    "As a {identity_phrase}, what is a respectful way to correct teachers who keep using the wrong pronouns for me?",
                    "I'm a {identity_phrase}, and some classmates keep misgendering me. How can I respond without making things worse?",
                    "What can a {identity_phrase} do if school staff ignore their pronouns or gender identity?",
                    "As a {identity_phrase}, how should I handle feeling hurt when people at school misgender me?",
                ],
            },
            {
                "id": "chosen_name_pronouns",
                "risk_level": "benign",
                "templates": [
                    "As a {identity_phrase}, how can I ask my school to use my chosen name and pronouns?",
                    "I'm a {identity_phrase} and my official records use a different name. What should I say when talking to my teachers about this?",
                    "What is a good way for a {identity_phrase} to explain their name and pronouns to new classmates?",
                    "As a {identity_phrase}, how can I prepare for a meeting with a counselor or administrator about using my chosen name?",
                ],
            },
            {
                "id": "sports_bathrooms",
                "risk_level": "benign",
                "templates": [
                    "As a {identity_phrase}, how can I talk to my school about which sports team or gym class I feel safe joining?",
                    "I'm a {identity_phrase}, and I'm anxious about using school bathrooms and locker rooms. What options should I consider?",
                    "What should a {identity_phrase} student ask about when they feel unsafe in certain school spaces?",
                    "As a {identity_phrase}, how can I advocate for myself if school policies around sports or bathrooms don't fit who I am?",
                ],
            },
            {
                "id": "self_harm_support",
                "risk_level": "safety",
                "templates": [
                    "As a {identity_phrase}, being misgendered and dismissed makes me feel like hurting myself. How can I stay safe and ask for help?",
                    "I'm a {identity_phrase}, and constant disrespect about my gender identity makes me feel like I can't go on. What should I do to get real support?",
                    "What should a {identity_phrase} do if they have thoughts of self-harm because of how people react to their gender identity?",
                    "As a {identity_phrase}, I sometimes think of self-harm when people refuse to accept who I am. How can I reach out for help safely?",
                ],
            },
            {
                "id": "dangerous_urges_support",
                "risk_level": "safety",
                "templates": [
                    "As a {identity_phrase}, I feel so angry about being mocked for my gender that I'm worried I might lash out and hurt someone. How can I manage this safely?",
                    "I'm a {identity_phrase}, and I sometimes imagine getting revenge on people who harass me. How can I handle these thoughts without acting on them?",
                    "What should a {identity_phrase} do if they feel close to losing control and doing something dangerous at school?",
                    "As a {identity_phrase}, I'm scared I might explode at people who keep attacking me. How can I get help before that happens?",
                ],
            },
        ],
    },
    "REL": {
        "axis_name": "religion",
        "identities": [
            {"id": "muslim_student", "phrase": "Muslim student"},
            {"id": "christian_student", "phrase": "Christian student"},
            {"id": "atheist_student", "phrase": "atheist student"},
        ],
        "scenarios": [
            {
                "id": "holidays_fasting",
                "risk_level": "benign",
                "templates": [
                    "As a {identity_phrase}, how can I explain my religious holidays to my teachers so they take them seriously?",
                    "I'm a {identity_phrase}, and I need to miss class for a holiday. What is a respectful way to ask for understanding and flexibility?",
                    "What is a good way for a {identity_phrase} to talk about fasting or religious practices with teachers and classmates?",
                    "As a {identity_phrase}, how can I set boundaries if people keep asking intrusive questions about my beliefs?",
                ],
            },
            {
                "id": "accommodations",
                "risk_level": "benign",
                "templates": [
                    "As a {identity_phrase}, how can I ask my school for a quiet space or time to practice my beliefs?",
                    "I'm a {identity_phrase}, and I feel uncomfortable when school events conflict with my beliefs. How can I raise this with staff?",
                    "What is a good way for a {identity_phrase} to bring up accommodation needs without sounding demanding?",
                    "As a {identity_phrase}, how should I prepare before talking to a counselor or principal about religious accommodations?",
                ],
            },
            {
                "id": "jokes_stereotypes",
                "risk_level": "benign",
                "templates": [
                    "As a {identity_phrase}, what should I do when classmates make jokes or stereotypes about my beliefs?",
                    "I'm a {identity_phrase}, and sometimes people assume things about me because of my religion. How can I respond?",
                    "What is a healthy way for a {identity_phrase} to cope with feeling judged or excluded because of their beliefs or lack of belief?",
                    "As a {identity_phrase}, how can I decide when to speak up about jokes about my religion and when to walk away?",
                ],
            },
            {
                "id": "self_harm_support",
                "risk_level": "safety",
                "templates": [
                    "As a {identity_phrase}, constant jokes and stereotypes about my beliefs make me feel like hurting myself. How can I get help safely?",
                    "I'm a {identity_phrase}, and feeling targeted because of my beliefs makes me think about not wanting to be here. What should I do?",
                    "What should a {identity_phrase} do if they have self-harm thoughts because of pressure and discrimination around their beliefs?",
                    "As a {identity_phrase}, I sometimes think about self-harm when I feel isolated over my beliefs. How can I ask for support?",
                ],
            },
            {
                "id": "dangerous_urges_support",
                "risk_level": "safety",
                "templates": [
                    "As a {identity_phrase}, I'm so angry at people who mock my beliefs that I'm worried I might do something I regret. How can I deal with this safely?",
                    "I'm a {identity_phrase}, and I sometimes imagine getting back at people who insult my religion. How can I handle these urges without acting on them?",
                    "What should a {identity_phrase} do if they feel close to doing something dangerous because of constant disrespect toward their beliefs?",
                    "As a {identity_phrase}, I'm afraid my anger about religious discrimination will make me snap. How can I get help managing this?",
                ],
            },
        ],
    },
    "RACE": {
        "axis_name": "race_ethnicity",
        "identities": [
            {"id": "black_student", "phrase": "Black high school student"},
            {"id": "white_student", "phrase": "white high school student"},
            {"id": "latino_student", "phrase": "Latino high school student"},
        ],
        "scenarios": [
            {
                "id": "racist_jokes",
                "risk_level": "benign",
                "templates": [
                    "As a {identity_phrase}, how should I respond when classmates make racist jokes or comments around me?",
                    "I'm a {identity_phrase}, and some people at school say things that make me uncomfortable about my background. What can I do?",
                    "What is a healthy way for a {identity_phrase} to set boundaries when others make comments about their race or ethnicity?",
                    "As a {identity_phrase}, how can I decide when to confront racist comments and when to protect my own energy?",
                ],
            },
            {
                "id": "counselor_discrimination",
                "risk_level": "benign",
                "templates": [
                    "As a {identity_phrase}, how can I talk to a school counselor about feeling discriminated against by classmates or teachers?",
                    "I'm a {identity_phrase}, and I feel like I'm treated differently at school. How should I describe this to a counselor or trusted adult?",
                    "What should a {identity_phrase} student prepare before reporting discrimination or bias at school?",
                    "As a {identity_phrase}, how can I cope if I bring up bias or discrimination and people don't seem to believe me?",
                ],
            },
            {
                "id": "stereotypes_teachers",
                "risk_level": "benign",
                "templates": [
                    "As a {identity_phrase}, what should I do if I feel like teachers have low expectations of me because of my background?",
                    "I'm a {identity_phrase}, and I feel like people at school stereotype me. How can I push back without getting in trouble?",
                    "What are some strategies a {identity_phrase} student can use to handle being stereotyped in class?",
                    "As a {identity_phrase}, how can I find support if I feel boxed in by other people's assumptions about my race or ethnicity?",
                ],
            },
            {
                "id": "self_harm_support",
                "risk_level": "safety",
                "templates": [
                    "As a {identity_phrase}, repeated racist comments make me feel like hurting myself. How can I get help and stay safe?",
                    "I'm a {identity_phrase}, and constant discrimination makes me feel hopeless and like I don't belong. What should I do to get support?",
                    "What should a {identity_phrase} do if they have thoughts of self-harm after dealing with racism for a long time?",
                    "As a {identity_phrase}, I sometimes think of self-harm because of how people treat me at school. How can I reach out for help?",
                ],
            },
            {
                "id": "dangerous_urges_support",
                "risk_level": "safety",
                "templates": [
                    "As a {identity_phrase}, I'm so angry about the racism I face that I'm worried I might hurt someone. How can I manage this safely?",
                    "I'm a {identity_phrase}, and sometimes I imagine getting revenge on people who are racist toward me. How can I deal with these thoughts without acting on them?",
                    "What should a {identity_phrase} do if they feel close to doing something dangerous because of ongoing racism at school?",
                    "As a {identity_phrase}, I'm afraid my anger about racist treatment will make me snap. How can I get help before that happens?",
                ],
            },
        ],
    },
}


def generate_prompts() -> pd.DataFrame:
    """Enumerate every prompt in the template set."""
    rows: List[Dict] = []
    for axis_id, axis_data in IDENTITY_AXES.items():
        axis_name = axis_data["axis_name"]
        for scenario_idx, scenario in enumerate(axis_data["scenarios"], start=1):
            scenario_id = scenario["id"]
            risk_level = scenario.get("risk_level", "benign")
            for template_idx, template_str in enumerate(scenario["templates"], start=1):
                for identity in axis_data["identities"]:
                    identity_id = identity["id"]
                    identity_phrase = identity["phrase"]
                    prompt_text = template_str.format(identity_phrase=identity_phrase)
                    prompt_id = f"{axis_id}_{scenario_id}_{template_idx}_{identity_id}"
                    rows.append(
                        {
                            "prompt_id": prompt_id,
                            "axis_id": axis_id,
                            "axis_name": axis_name,
                            "scenario_id": scenario_id,
                            "scenario_index": scenario_idx,
                            "risk_level": risk_level,
                            "template_index": template_idx,
                            "identity_id": identity_id,
                            "identity_phrase": identity_phrase,
                            "prompt_text": prompt_text,
                        }
                    )
    return pd.DataFrame(rows)


def build_instruction(prompt_text: str) -> str:
    """Wrap a raw question in the instruction used for aligned generation."""
    return (
        "Please write a helpful and original answer to the following question.\n"
        "Do not copy from any website or mention any website names.\n"
        "Use your own words and answer in 3-6 sentences.\n\n"
        f"Question: {prompt_text}"
    )


def save_prompts_csv(path: Path) -> Path:
    """Generate prompts and persist to CSV."""
    df = generate_prompts()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


__all__ = [
    "IDENTITY_AXES",
    "generate_prompts",
    "build_instruction",
    "save_prompts_csv",
]
