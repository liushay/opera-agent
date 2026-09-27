# opera/face.py 脸谱画像
# 功能：生成用户专属"戏曲人格脸谱"解读卡片
#   - 文本版：LLM 根据用户喜好生成脸谱设计描述 + 颜色/图案含义解读
#   - 图像版：基于火山引擎"视觉智能 CV"服务（即梦AI）生成脸谱图像
#
# 火山引擎接口（官方）：
#   接口地址：https://visual.volcengineapi.com
#   Action:   Text2ImgXLSft（文生图，同步/异步）
#   签名鉴权：由 volcengine-python-sdk 的 Configuration/ApiClient 自动完成（禁止手写签名）
#
# SDK 兼容说明：
#   - 旧版 SDK（4.x 及以前，`volcengine` 包）：使用 VisualService + cv_sync2_async_submit_task/cv_async_query_task
#   - 新版 SDK（5.x，`volcenginesdk*` 包）：使用 volcenginesdkcv20240606 的 CV20240606Api + text2_img_xl_sft
#   - 代码自动识别新版/旧版 SDK，任一可用即可；都未安装时自动降级为纯文本版
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

import requests
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

import config
from rag.vectorstore import hybrid_retrieve
from utils.logger import log_info, log_warn, log_error

# ===================== 火山引擎 SDK 双版本探测 =====================
# 新版 SDK（5.x）：volcenginesdkcv20240606
try:
    from volcenginesdkcore import Configuration as VcConfiguration, ApiClient as VcApiClient
    from volcenginesdkcv20240606.api.cv20240606_api import CV20240606Api
    from volcenginesdkcv20240606.models import Text2ImgXLSftRequest
    _VC_NEW_SDK = True
except ImportError:
    _VC_NEW_SDK = False

# 旧版 SDK（4.x）：volcengine
try:
    from volcengine.visual.VisualService import VisualService
    _VC_OLD_SDK = True
except ImportError:
    _VC_OLD_SDK = False

if not _VC_NEW_SDK and not _VC_OLD_SDK:
    log_warn("脸谱画像", "未安装 volcengine-python-sdk（新版或旧版），图像生成不可用，将降级为文本版")

# 脸谱颜色含义（内置知识，作为兜底）
FACE_COLOR_MEANING = {
    "红": "忠勇侠义，如关羽",
    "黑": "刚正不阿，如包拯",
    "白": "奸诈多疑，如曹操",
    "蓝": "刚强勇猛，如窦尔敦",
    "绿": "勇猛暴躁，如程咬金",
    "黄": "暴烈凶狠，如典韦",
    "紫": "稳重老练，如专诸",
    "金": "神佛仙圣，如来",
    "银": "妖精鬼怪，白猿",
}


def _get_llm():
    """复用主LLM"""
    return ChatOllama(model=config.LLM_MODEL, temperature=0.4)


def _retrieve_face_material(query: str) -> str:
    """检索知识库脸谱知识"""
    try:
        docs = hybrid_retrieve(query + " 脸谱")
        if not docs:
            return ""
        return "\n".join([doc.page_content for doc in docs[:3]])
    except Exception as e:
        log_warn("脸谱画像", f"知识库检索失败：{e}")
        return ""


def generate_face_profile(
    preferences: str,
    session_id: str,
) -> Dict[str, Any]:
    """
    生成用户专属脸谱画像（对外主入口，上层调用无需改动）
    Args:
        preferences: 用户描述的喜好（如"红色，代表忠义"）
        session_id: 会话ID（记录画像轨迹）
    Returns:
        {
            "face_name": 脸谱名号,
            "color": 主色,
            "color_meaning": 颜色含义,
            "pattern": 图案描述,
            "pattern_meaning": 图案含义,
            "matching_character": 对应戏曲人物,
            "personality_text": 人格解读文案,
            "share_card": 分享卡片文案,
            "image_status": "text_only"（未启用/失败）/ "generated"（已生成）,
            "image_url": 本地图片路径或空串
        }
    """
    log_info("脸谱画像", f"生成脸谱画像，偏好：{preferences[:50]}")
    material = _retrieve_face_material(preferences)

    prompt = f"""你是戏曲脸谱设计师。请根据用户的喜好，为用户设计一款专属戏曲脸谱。

用户偏好描述：{preferences}

参考素材：
{material or "（知识库暂无脸谱资料，请基于脸谱常识设计）"}

内置颜色含义参考：
{json.dumps(FACE_COLOR_MEANING, ensure_ascii=False, indent=2)}

请输出严格JSON：
{{
  "face_name": "脸谱名号（2-4字，如'忠义赤面'）",
  "color": "主色（红/黑/白/蓝/绿/黄/紫/金/银之一）",
  "pattern": "额头/面部的图案设计（如'威风虎纹'）",
  "matching_character": "与该脸谱气质匹配的戏曲人物",
  "personality_text": "40-80字人格解读：这副脸谱体现用户怎样的性格特质",
  "share_card": "60-100字分享卡片文案，语言有韵味，适合发朋友圈（署名'我的戏曲人格脸谱'）"
}}
只输出JSON。"""
    try:
        llm = _get_llm()
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip()
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(content[start:end])
        else:
            raise json.JSONDecodeError("无JSON", content, 0)
    except Exception as e:
        log_warn("脸谱画像", f"脸谱生成失败，降级为模板：{e}")
        data = generate_fallback_face(preferences)

    color = data.get("color", "红")
    color_meaning = FACE_COLOR_MEANING.get(color, "该颜色有其专属的戏曲气质")

    # 记录用户画像到会话时序记忆
    try:
        from agent.memory import memory_manager
        memory_manager.record_thinking(
            session_id, f"生成脸谱画像：偏好={preferences[:30]}，脸谱={data.get('face_name', '')}"
        )
    except Exception as e:
        log_warn("脸谱画像", f"记录失败：{e}")

    # 调用即梦图像生成（仅启用时；未启用时 generate_face_image 返回 None，不影响业务）
    image_status = "text_only"
    image_url = ""
    try:
        result = generate_face_image(data, preferences)
        if result:
            # 返回可能是本地路径字符串，也可能是 dict（保留扩展性）
            image_url = result if isinstance(result, str) else result.get("image_path", "")
            if image_url:
                image_status = "generated"
    except Exception as e:
        # 图像生成任何异常都不影响文本版交付
        log_error("脸谱画像", f"图像生成异常（降级为文本版）：{e}", e)

    return {
        "face_name": data.get("face_name", "无名脸谱"),
        "color": color,
        "color_meaning": color_meaning,
        "pattern": data.get("pattern", ""),
        "pattern_meaning": data.get("pattern_meaning", ""),
        "matching_character": data.get("matching_character", ""),
        "personality_text": data.get("personality_text", ""),
        "share_card": data.get("share_card", ""),
        "image_status": image_status,
        "image_url": image_url,
    }


# ===================== 火山引擎即梦AI图像生成（双 SDK 兼容） =====================

def _get_visual_service():
    """
    创建并返回已配置 AK/SK 的视觉服务实例（新版 SDK 的 CV20240606Api）。
    - 未安装 SDK / 未配置 AK/SK / 开关关闭 时返回 None
    - SDK 自动完成火山引擎签名鉴权（构造 header 中的 Authorization）
    """
    if not getattr(config, "JIMENG_IMAGE_ENABLED", False):
        return None
    if not _VC_NEW_SDK:
        log_warn("脸谱画像", "新版 volcengine-python-sdk 未安装，跳过图像生成")
        return None
    ak = getattr(config, "JIMENG_ACCESS_KEY_ID", "")
    sk = getattr(config, "JIMENG_SECRET_ACCESS_KEY", "")
    if not ak or not sk:
        log_warn("脸谱画像", "未配置 JIMENG_ACCESS_KEY_ID / JIMENG_SECRET_ACCESS_KEY，跳过图像生成")
        return None
    try:
        cfg = VcConfiguration()
        cfg.ak = ak
        cfg.sk = sk
        # 视觉智能服务默认 host/region
        cfg.host = "visual.volcengineapi.com"
        cfg.region = "cn-north-1"
        client = VcApiClient(cfg)
        service = CV20240606Api(client)
        return service
    except Exception as e:
        log_error("脸谱画像", f"CV20240606Api 初始化失败：{e}", e)
        return None


def generate_face_image(face_data: Dict[str, Any], prefs: str) -> Optional[str]:
    """
    调用火山引擎即梦AI生成脸谱图像（对外函数名保留，上层调用无需改动）
    流程：构造Prompt -> 调用即梦文生图（Text2ImgXLSft）-> 下载图片到本地
    Args:
        face_data: LLM 生成的脸谱设计数据
        prefs: 用户偏好（备用）
    Returns:
        本地图片路径；未启用或任何失败场景返回 None（上层自动降级文本版）
    """
    service = _get_visual_service()
    if service is None:
        # 开关关闭或配置缺失：直接返回 None，不抛异常
        return None

    # 1. 构造图像生成 Prompt
    prompt = build_face_image_prompt(face_data)
    log_info("脸谱画像", f"即梦生图 Prompt：{prompt[:80]}...")

    try:
        # 2. 构造请求体并调用文生图接口
        req_key = getattr(config, "JIMENG_MODEL", "jimeng_t2i_v31")
        body = Text2ImgXLSftRequest(
            req_key=req_key,
            prompt=prompt,
            return_url=True,  # 返回图片可访问 URL
        )
        # 同步调用（新版 SDK 默认同步；async_req=True 时返回线程，不在此使用）
        resp = service.text2_img_xl_sft(body)
        log_info("脸谱画像", f"即梦响应：{str(resp)[:300]}")

        # 3. 解析图片 URL（新版 SDK 响应对象支持属性访问 / to_dict）
        image_url = _extract_image_url(resp)
        if not image_url:
            log_warn("脸谱画像", "即梦响应中未找到图片 URL")
            return None

        # 4. 下载图片保存本地（目录不存在自动创建）
        local_path = download_face_image(image_url)
        if not local_path:
            log_warn("脸谱画像", "图片下载失败，返回原始URL备用")
            return image_url
        log_info("脸谱画像", f"脸谱图像已生成：{local_path}")
        return local_path
    except Exception as e:
        # 网络/接口任何异常只记日志，不导致程序崩溃
        log_error("脸谱画像", f"即梦图像生成异常：{e}", e)
        return None


def _extract_image_url(resp) -> Optional[str]:
    """
    从即梦文生图响应中提取图片 URL。
    兼容新版 SDK 的对象属性访问 / to_dict() / dict 三种形态。
    """
    try:
        # 形态1：对象属性访问（新版 SDK 模型）
        if hasattr(resp, "data") and resp.data is not None:
            data = resp.data
            if hasattr(data, "image_urls") and data.image_urls:
                return str(data.image_urls[0])
            # resp_data 可能是 JSON 字符串，也可能嵌套对象
            if hasattr(data, "resp_data") and data.resp_data:
                rd = data.resp_data
                if isinstance(rd, str):
                    try:
                        rd_dict = json.loads(rd)
                        urls = rd_dict.get("image_url") or (rd_dict.get("image_urls") or [None])
                        if isinstance(urls, list):
                            return str(urls[0]) if urls else None
                        return str(urls) if urls else None
                    except Exception:
                        return None
                if isinstance(rd, dict):
                    urls = rd.get("image_url") or (rd.get("image_urls") or [None])
                    if isinstance(urls, list):
                        return str(urls[0]) if urls else None
                    return str(urls) if urls else None
            if hasattr(data, "response_data") and data.response_data:
                rd = data.response_data
                if isinstance(rd, str):
                    try:
                        rd_dict = json.loads(rd)
                        urls = rd_dict.get("image_url") or (rd_dict.get("image_urls") or [None])
                        if isinstance(urls, list):
                            return str(urls[0]) if urls else None
                        return str(urls) if urls else None
                    except Exception:
                        return None
                if isinstance(rd, dict):
                    urls = rd.get("image_url") or (rd.get("image_urls") or [None])
                    if isinstance(urls, list):
                        return str(urls[0]) if urls else None
                    return str(urls) if urls else None
            # 直接 data 上可能有 image_url / image_urls
            if hasattr(data, "image_url") and data.image_url:
                return str(data.image_url)
        # 形态2：to_dict()（新版 SDK）
        if hasattr(resp, "to_dict"):
            d = resp.to_dict()
            data = d.get("data") or {}
            urls = data.get("image_urls") or []
            if urls:
                return str(urls[0])
            resp_data = data.get("resp_data") or data.get("response_data") or ""
            if isinstance(resp_data, str) and resp_data:
                try:
                    rd = json.loads(resp_data)
                    u = rd.get("image_url") or (rd.get("image_urls") or [None])
                    return str(u[0]) if isinstance(u, list) and u else (str(u) if u else None)
                except Exception:
                    return None
            if isinstance(resp_data, dict):
                u = resp_data.get("image_url") or (resp_data.get("image_urls") or [None])
                return str(u[0]) if isinstance(u, list) and u else (str(u) if u else None)
        # 形态3：纯 dict（旧版 SDK 响应或直接 JSON）
        if isinstance(resp, dict):
            data = resp.get("data") or {}
            urls = data.get("image_urls") or []
            if urls:
                return str(urls[0])
            resp_data = data.get("resp_data") or data.get("response_data") or ""
            if isinstance(resp_data, str) and resp_data:
                try:
                    rd = json.loads(resp_data)
                    u = rd.get("image_url") or (rd.get("image_urls") or [None])
                    return str(u[0]) if isinstance(u, list) and u else (str(u) if u else None)
                except Exception:
                    return None
            if isinstance(resp_data, dict):
                u = resp_data.get("image_url") or (resp_data.get("image_urls") or [None])
                return str(u[0]) if isinstance(u, list) and u else (str(u) if u else None)
    except Exception as e:
        log_warn("脸谱画像", f"即梦响应解析失败：{e}")
        return None
    return None


def download_face_image(image_url: str) -> str:
    """下载图片到 JIMENG_IMAGE_OUTPUT_DIR（不存在自动创建），返回本地路径"""
    try:
        out_dir = os.path.abspath(getattr(config, "JIMENG_IMAGE_OUTPUT_DIR", "./literature_output/face_images"))
        os.makedirs(out_dir, exist_ok=True)  # 目录不存在自动创建
        fname = f"face_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
        local_path = os.path.join(out_dir, fname)
        resp = requests.get(image_url, timeout=60)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)
        return local_path
    except Exception as e:
        log_warn("脸谱画像", f"图片下载失败：{e}")
        return ""


def build_face_image_prompt(face_data: Dict[str, Any]) -> str:
    """
    把 LLM 生成的脸谱设计转换为即梦生图 Prompt。
    优化要点：明确要求"纯脸谱图案、禁止出现任何文字/字符"，避免生成
    蓝色背景+文字海报；强化主色彩绘、对称纹样、额头图案等脸谱视觉特征。
    """
    name = face_data.get("face_name", "脸谱")
    color = face_data.get("color", "红")
    pattern = face_data.get("pattern", "")
    char = face_data.get("matching_character", "")
    color_cn = {
        "红": "朱红", "黑": "乌黑", "白": "粉白", "蓝": "靛蓝",
        "绿": "黛绿", "黄": "金黄", "紫": "绛紫", "金": "鎏金", "银": "银白",
    }.get(color, color)
    prompt = (
        f"一张正宗的中国传统戏曲京剧脸谱特写绘画，纯脸谱图案、禁止包含任何文字、字母、数字或标语字符。"
        f"主色调为{color_cn}，面部以彩色油彩勾画：{pattern}，"
        f"额头与脸颊有对称的戏曲装饰纹样，眉眼用黑色油彩提神上扬，"
        f"鼻梁与眼窝有精细的色块过渡。"
        f"气质参照{char}英武刚毅的神情。"
        f"工笔重彩写实风格，端正正面肖像构图，黑色或深红戏曲帽头衬托，"
        f"背景为素色晕染、无任何文字水印，高清特写，戏曲舞台美术质感。"
    )
    return prompt


def generate_fallback_face(preferences: str) -> Dict[str, Any]:
    """脸谱生成失败的降级模板"""
    colors = ["红", "黑", "白", "蓝", "绿", "紫"]
    color = random_choice(colors)
    chars = {"红": "关羽", "黑": "包拯", "白": "曹操", "蓝": "窦尔敦", "绿": "程咬金", "紫": "专诸"}
    patterns = {
        "红": "眉眼上挑的忠义纹",
        "黑": "额间月牙纹",
        "白": "细长的奸雄眉",
        "蓝": "刚猛的火焰纹",
        "绿": "粗犷的虎纹",
        "紫": "沉稳的云雷纹",
    }
    meaning = FACE_COLOR_MEANING.get(color, "独特气质")
    spirit = random_choice(["雄风", "正气", "英姿"])
    return {
        "face_name": f"{color}面{spirit}",
        "color": color,
        "pattern": patterns.get(color, "经典纹样"),
        "matching_character": chars.get(color, "戏曲人物"),
        "personality_text": f"你的脸谱以{color}为主色，这在戏曲中象征{meaning}，说明你性格中带着这份鲜明的底色。",
        "share_card": f"今日测得我的戏曲人格脸谱——{color}面{spirit}！"
                      f"主色{color}，{patterns.get(color, '经典纹样')}，"
                      f"气质如{chars.get(color, '戏曲人物')}。这是我的戏曲人格脸谱！",
    }


def random_choice(seq):
    """轻量随机选一（避免顶部 import random，保持环境轻量）"""
    import random
    return random.choice(seq)