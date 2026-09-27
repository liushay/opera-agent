# api/routes/opera_routes.py 戏曲科普功能接口
# 五大功能：戏词解剖 / 人物对谈 / 知识闯关 / 学戏路线 / 脸谱画像
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional
import os

from api.schema import CommonResponse
from opera import (
    annotate_lyrics,
    get_character_profile,
    character_chat,
    CHARACTER_LIST,
    generate_quiz,
    check_answer,
    QUIZ_TOPICS,
    generate_course,
    get_course_progress,
    generate_face_profile,
)
from utils.logger import log_error
from utils.logger import log_info

router = APIRouter(prefix="/api/opera", tags=["戏曲科普"])


# ===================== 请求体 =====================

class LyricsRequest(BaseModel):
    lyrics: str = Field(..., description="戏词原文")


class CharacterChatRequest(BaseModel):
    character_id: str = Field(..., description="人物ID")
    message: str = Field(..., description="用户发言")
    session_id: str = Field("opera_default_session", description="会话ID")


class QuizGenerateRequest(BaseModel):
    topic: Optional[str] = Field("混合", description="主题，可选：混合或五类主题")
    difficulty: Optional[str] = Field("入门", description="难度：小白/入门/票友/老戏骨")
    session_id: str = Field("opera_default_session", description="会话ID")


class QuizCheckRequest(BaseModel):
    question_id: str = Field(..., description="题目ID")
    user_answer: str = Field(..., description="用户选项 A/B/C/D")
    correct_answer: str = Field(..., description="正确答案")
    explanation: str = Field("", description="知识点讲解")
    session_id: str = Field("opera_default_session", description="会话ID")


class CourseGenerateRequest(BaseModel):
    topic: str = Field(..., description="想学的剧种，如京剧、昆曲")
    days: Optional[int] = Field(7, description="课程天数（1-14）")
    level: Optional[str] = Field("入门", description="用户水平")
    session_id: str = Field("opera_default_session", description="会话ID")


class CourseProgressRequest(BaseModel):
    session_id: str = Field("opera_default_session", description="会话ID")
    topic: Optional[str] = Field("", description="指定主题（可选）")


class FaceGenerateRequest(BaseModel):
    preferences: str = Field(..., description="用户喜好描述，如'红色，代表忠义'")
    session_id: str = Field("opera_default_session", description="会话ID")


# ===================== 五大功能接口 =====================

@router.post("/lyrics/annotate", response_model=CommonResponse)
async def opera_lyrics_annotate(req: LyricsRequest):
    """戏词解剖：逐句翻译/典故/心境/唱腔/品鉴"""
    try:
        result = annotate_lyrics(req.lyrics)
        if "error" in result:
            return CommonResponse(code=500, msg=result["error"], data={})
        return CommonResponse(code=200, msg="戏词解剖完成", data=result)
    except Exception as e:
        log_error("戏词解剖接口", str(e), e)
        return CommonResponse(code=500, msg=f"戏词解剖失败：{str(e)}", data={})


@router.get("/character/list", response_model=CommonResponse)
async def opera_character_list():
    """人物列表"""
    try:
        return CommonResponse(
            code=200,
            msg="人物列表获取成功",
            data={"characters": CHARACTER_LIST},
        )
    except Exception as e:
        return CommonResponse(code=500, msg=str(e), data={})


@router.post("/character/chat", response_model=CommonResponse)
async def opera_character_chat(req: CharacterChatRequest):
    """戏中人对谈"""
    try:
        result = character_chat(req.character_id, req.message, req.session_id)
        if "error" in result:
            return CommonResponse(code=500, msg=result["error"], data={})
        return CommonResponse(code=200, msg="对谈成功", data=result)
    except Exception as e:
        log_error("戏中人对谈接口", str(e), e)
        return CommonResponse(code=500, msg=f"对谈失败：{str(e)}", data={})


@router.post("/quiz/generate", response_model=CommonResponse)
async def opera_quiz_generate(req: QuizGenerateRequest):
    """知识闯关：出题"""
    try:
        result = generate_quiz(req.topic, req.difficulty, req.session_id)
        if "error" in result:
            return CommonResponse(code=500, msg=result["error"], data={})
        return CommonResponse(code=200, msg="出题成功", data=result)
    except Exception as e:
        log_error("知识闯关出题接口", str(e), e)
        return CommonResponse(code=500, msg=f"出题失败：{str(e)}", data={})


@router.post("/quiz/topics", response_model=CommonResponse)
async def opera_quiz_topics():
    """知识闯关：可用主题"""
    try:
        return CommonResponse(code=200, msg="主题列表获取成功", data={"topics": QUIZ_TOPICS})
    except Exception as e:
        return CommonResponse(code=500, msg=str(e), data={})


@router.post("/quiz/check", response_model=CommonResponse)
async def opera_quiz_check(req: QuizCheckRequest):
    """知识闯关：判题"""
    try:
        result = check_answer(
            req.question_id, req.user_answer,
            req.correct_answer, req.explanation, req.session_id,
        )
        return CommonResponse(code=200, msg="判题完成", data=result)
    except Exception as e:
        log_error("知识闯关判题接口", str(e), e)
        return CommonResponse(code=500, msg=f"判题失败：{str(e)}", data={})


@router.post("/course/generate", response_model=CommonResponse)
async def opera_course_generate(req: CourseGenerateRequest):
    """个性化学戏路线：生成课程"""
    try:
        result = generate_course(req.topic, req.days, req.session_id, req.level)
        if "error" in result:
            return CommonResponse(code=500, msg=result["error"], data={})
        return CommonResponse(code=200, msg="课程生成成功", data=result)
    except Exception as e:
        log_error("学戏路线生成接口", str(e), e)
        return CommonResponse(code=500, msg=f"课程生成失败：{str(e)}", data={})


@router.post("/course/progress", response_model=CommonResponse)
async def opera_course_progress(req: CourseProgressRequest):
    """个性化学戏路线：查询进度"""
    try:
        result = get_course_progress(req.session_id, req.topic)
        return CommonResponse(code=200, msg="进度查询成功", data=result)
    except Exception as e:
        log_error("学戏进度查询接口", str(e), e)
        return CommonResponse(code=500, msg=f"进度查询失败：{str(e)}", data={})


@router.post("/face/generate", response_model=CommonResponse)
async def opera_face_generate(req: FaceGenerateRequest):
    """脸谱画像：生成专属脸谱解读"""
    try:
        result = generate_face_profile(req.preferences, req.session_id)
        if "error" in result:
            return CommonResponse(code=500, msg=result["error"], data={})
        # 图片字段处理：真实本地磁盘路径放入 local_file_path（供前端直接读取图片文件），
        # 同时将 /static URL 放入 image_url（供前端通过后端静态服务展示）。
        # 兼容前端可能访问 local_file_path / image_url / real_file_path 任一字段。
        image_url = result.get("image_url", "")
        local_file_path = ""
        if image_url and not image_url.startswith(("http://", "https://")):
            # image_url 此时是本地磁盘路径（绝对或相对）
            norm = image_url.replace("\\", "/")
            local_file_path = norm
            # 计算 /static 相对路径
            if "/literature_output/" in norm:
                rel = norm.split("/literature_output/", 1)[1]
            elif norm.startswith("./literature_output/"):
                rel = norm[len("./literature_output/"):]
            else:
                import os as _os
                rel = _os.path.basename(norm)
            result["image_url"] = f"/static/{rel}"
        # 总是补充 local_file_path / real_file_path 字段（即使为空字符串，避免前端 KeyError）
        result["local_file_path"] = local_file_path
        result["real_file_path"] = local_file_path
        return CommonResponse(code=200, msg="脸谱画像生成成功", data=result)
    except Exception as e:
        log_error("脸谱画像接口", str(e), e)
        return CommonResponse(code=500, msg=f"脸谱生成失败：{str(e)}", data={})
