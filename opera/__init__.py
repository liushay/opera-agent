# opera/__init__.py 戏曲科普业务模块
# 包含五大用户向功能：
#   - lyrics.py      戏词解剖室（逐句译注/典故/唱腔/心境）
#   - character.py   戏中人对谈（角色人格化对话）
#   - quiz.py        知识闯关（游戏化出题/判题）
#   - guide.py       个性化学戏路线（阶梯课程/进度追踪）
#   - face.py        脸谱画像（脸谱解读/预留图像API）
from opera.lyrics import annotate_lyrics
from opera.character import get_character_profile, character_chat, CHARACTER_LIST
from opera.quiz import generate_quiz, generate_quiz_batch, check_answer, QUIZ_TOPICS
from opera.guide import generate_course, get_course_progress
from opera.face import generate_face_profile

__all__ = [
    "annotate_lyrics",
    "get_character_profile",
    "character_chat",
    "CHARACTER_LIST",
    "generate_quiz",
    "generate_quiz_batch",
    "check_answer",
    "QUIZ_TOPICS",
    "generate_course",
    "get_course_progress",
    "generate_face_profile",
]
