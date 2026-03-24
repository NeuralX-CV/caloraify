#!/usr/bin/env python3
# =============================================================================
# CaLoRAify Telegram Bot
# Features: photo analysis, meal logging, daily calorie tracking, streaks
#
# SETUP:
#   1. Create a bot via @BotFather on Telegram → get BOT_TOKEN
#   2. Deploy the HuggingFace Space → get SPACE_URL
#   3. pip install -r requirements_bot.txt
#   4. Set environment variables (or edit the CONFIG section below):
#        TELEGRAM_BOT_TOKEN = "123456:ABC..."
#        CALORAIFY_API_URL  = "https://your_username-caloraify.hf.space"
#        CALORAIFY_API_KEY  = "your_secret_api_key"
#   5. python telegram_bot.py
# =============================================================================

import os
import io
import json
import base64
import logging
import sqlite3
import asyncio
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
BOT_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
API_URL     = os.environ.get("CALORAIFY_API_URL",  "https://unnatrathi-caloraify.hf.space")
API_KEY     = os.environ.get("CALORAIFY_API_KEY",  "")
DB_PATH     = Path(os.environ.get("DB_PATH", "caloraify.db"))
DAILY_GOAL  = int(os.environ.get("DAILY_CALORIE_GOAL", "2000"))  

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("caloraify_bot")


# ── DATABASE ──────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS meals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            username    TEXT,
            logged_at   TEXT NOT NULL,          -- ISO datetime
            log_date    TEXT NOT NULL,          -- YYYY-MM-DD for grouping
            ingredients TEXT,
            portions    TEXT,
            calories    REAL,
            protein_g   REAL,
            carbs_g     REAL,
            fat_g       REAL,
            fibre_g     REAL,
            raw_text    TEXT
        );

        CREATE TABLE IF NOT EXISTS user_settings (
            user_id     INTEGER PRIMARY KEY,
            username    TEXT,
            daily_goal  INTEGER DEFAULT 2000,
            timezone    TEXT DEFAULT 'UTC'
        );

        CREATE INDEX IF NOT EXISTS idx_meals_user_date ON meals(user_id, log_date);
    """)
    conn.commit()
    conn.close()


def get_conn():
    return sqlite3.connect(DB_PATH)


def log_meal(user_id: int, username: str, data: dict) -> int:
    now = datetime.utcnow()
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO meals
            (user_id, username, logged_at, log_date,
             ingredients, portions, calories, protein_g, carbs_g, fat_g, fibre_g, raw_text)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            user_id, username,
            now.isoformat(), now.strftime("%Y-%m-%d"),
            data.get("ingredients", ""),
            data.get("portion_notes", ""),
            data.get("calories"),
            data.get("protein_g"),
            data.get("carbs_g"),
            data.get("fat_g"),
            data.get("fibre_g"),
            data.get("raw_text", ""),
        ))
        return c.lastrowid


def get_daily_summary(user_id: int, day: str) -> dict:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT
                COUNT(*) as meals,
                ROUND(SUM(COALESCE(calories, 0)), 1) as total_cal,
                ROUND(SUM(COALESCE(protein_g, 0)), 1) as total_prot,
                ROUND(SUM(COALESCE(carbs_g, 0)),   1) as total_carbs,
                ROUND(SUM(COALESCE(fat_g, 0)),     1) as total_fat,
                ROUND(SUM(COALESCE(fibre_g, 0)),   1) as total_fibre
            FROM meals
            WHERE user_id=? AND log_date=?
        """, (user_id, day))
        row = c.execute("""SELECT * FROM meals WHERE user_id=? AND log_date=? ORDER BY logged_at""",
                        (user_id, day)).fetchall()
        totals = c.fetchone()

    meals_list = []
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, logged_at, ingredients, calories FROM meals "
            "WHERE user_id=? AND log_date=? ORDER BY logged_at",
            (user_id, day)
        ).fetchall()
        for r in rows:
            meals_list.append({
                "id": r[0], "time": r[1][11:16],
                "ingredients": r[2], "calories": r[3],
            })

    with get_conn() as conn:
        t = conn.execute("""
            SELECT
                ROUND(SUM(COALESCE(calories, 0)), 1),
                ROUND(SUM(COALESCE(protein_g, 0)), 1),
                ROUND(SUM(COALESCE(carbs_g, 0)), 1),
                ROUND(SUM(COALESCE(fat_g, 0)), 1),
                ROUND(SUM(COALESCE(fibre_g, 0)), 1),
                COUNT(*)
            FROM meals WHERE user_id=? AND log_date=?
        """, (user_id, day)).fetchone()

    return {
        "date":       day,
        "total_cal":  t[0] or 0,
        "protein_g":  t[1] or 0,
        "carbs_g":    t[2] or 0,
        "fat_g":      t[3] or 0,
        "fibre_g":    t[4] or 0,
        "meal_count": t[5] or 0,
        "meals":      meals_list,
    }


def get_user_goal(user_id: int) -> int:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT daily_goal FROM user_settings WHERE user_id=?", (user_id,)
        ).fetchone()
    return row[0] if row else DAILY_GOAL


def set_user_goal(user_id: int, username: str, goal: int):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO user_settings (user_id, username, daily_goal)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET daily_goal=excluded.daily_goal, username=excluded.username
        """, (user_id, username, goal))


def get_streak(user_id: int) -> int:
    """Count consecutive days (ending today or yesterday) with at least 1 logged meal."""
    today = date.today()
    streak = 0
    check  = today
    with get_conn() as conn:
        while True:
            day_str = check.strftime("%Y-%m-%d")
            count = conn.execute(
                "SELECT COUNT(*) FROM meals WHERE user_id=? AND log_date=?",
                (user_id, day_str)
            ).fetchone()[0]
            if count == 0:
                # Allow a 1-day gap only if today has no meals yet (haven't eaten today)
                if check == today and streak == 0:
                    check -= timedelta(days=1)
                    continue
                break
            streak += 1
            check  -= timedelta(days=1)
    return streak


def get_weekly_summary(user_id: int) -> list[dict]:
    today = date.today()
    days  = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]
    result = []
    with get_conn() as conn:
        for d in days:
            row = conn.execute("""
                SELECT ROUND(SUM(COALESCE(calories,0)),0), COUNT(*)
                FROM meals WHERE user_id=? AND log_date=?
            """, (user_id, d)).fetchone()
            result.append({"date": d, "calories": row[0] or 0, "meals": row[1] or 0})
    return result


def delete_meal(user_id: int, meal_id: int) -> bool:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM meals WHERE id=? AND user_id=?", (meal_id, user_id))
        return c.rowcount > 0


# ── INFERENCE CLIENT ──────────────────────────────────────────────────────────
async def analyze_food_image(image_bytes: bytes) -> dict:
    b64 = base64.b64encode(image_bytes).decode()
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{API_URL}/analyze",
            json={"image_b64": b64, "max_new_tokens": 300},
            headers={"x-api-key": API_KEY},
        )
    resp.raise_for_status()
    return resp.json()


# ── FORMATTERS ────────────────────────────────────────────────────────────────
def _bar(value: float, goal: float, width: int = 12) -> str:
    """Render a simple text progress bar."""
    pct    = min(value / goal, 1.0) if goal > 0 else 0
    filled = int(pct * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {int(pct*100)}%"


def format_analysis_message(data: dict, meal_id: int) -> str:
    cal  = data.get("calories")
    prot = data.get("protein_g")
    carb = data.get("carbs_g")
    fat  = data.get("fat_g")
    fib  = data.get("fibre_g")

    lines = ["🍽 *Meal Analysis*\n"]
    if data.get("ingredients"):
        lines.append(f"🥦 *Ingredients:* {data['ingredients']}")
    if data.get("portion_notes"):
        lines.append(f"⚖️ *Portions:* {data['portion_notes']}")
    lines.append("")
    if cal  is not None: lines.append(f"🔥 Calories: *{cal:.0f} kcal*")
    if prot is not None: lines.append(f"💪 Protein:  *{prot:.1f} g*")
    if carb is not None: lines.append(f"🌾 Carbs:    *{carb:.1f} g*")
    if fat  is not None: lines.append(f"🥑 Fat:      *{fat:.1f} g*")
    if fib  is not None: lines.append(f"🌿 Fibre:    *{fib:.1f} g*")
    lines.append(f"\n_Meal ID #{meal_id} — logged ✅_")
    return "\n".join(lines)


def format_daily_summary(summary: dict, goal: int) -> str:
    cal   = summary["total_cal"]
    prot  = summary["protein_g"]
    carb  = summary["carbs_g"]
    fat   = summary["fat_g"]
    fib   = summary["fibre_g"]
    meals = summary["meal_count"]
    day   = summary["date"]

    remaining = max(goal - cal, 0)
    bar       = _bar(cal, goal)

    lines = [
        f"📅 *Daily Summary — {day}*\n",
        f"🍽 Meals logged: *{meals}*",
        f"🔥 Calories: *{cal:.0f} / {goal} kcal*",
        f"{bar}",
        f"_Remaining: {remaining:.0f} kcal_\n",
        f"💪 Protein: *{prot:.1f} g*",
        f"🌾 Carbs:   *{carb:.1f} g*",
        f"🥑 Fat:     *{fat:.1f} g*",
        f"🌿 Fibre:   *{fib:.1f} g*",
    ]

    if summary["meals"]:
        lines.append("\n*Meal breakdown:*")
        for m in summary["meals"]:
            cal_str = f"{m['calories']:.0f} kcal" if m["calories"] else "unknown kcal"
            ing     = (m["ingredients"] or "")[:40]
            lines.append(f"  {m['time']} — {ing}… ({cal_str})")

    return "\n".join(lines)


def format_weekly_chart(data: list[dict], goal: int) -> str:
    lines = ["📊 *7-Day Calorie History*\n"]
    for d in data:
        day_label = d["date"][5:]        # MM-DD
        cal       = d["calories"]
        bar_len   = int(min(cal / goal, 1.5) * 15)
        bar       = "█" * bar_len
        over      = " ⚠️ over goal" if cal > goal else ""
        lines.append(f"`{day_label}` {bar} {cal:.0f}{over}")
    lines.append(f"\n_Goal: {goal} kcal/day_")
    return "\n".join(lines)


def streak_message(streak: int) -> str:
    if streak == 0:
        return "📭 No streak yet — log your first meal to start!"
    emojis = {1: "🌱", 2: "🌿", 3: "🌳", 7: "🔥", 14: "💪", 30: "🏆"}
    icon   = "🔥"
    for threshold, emoji in sorted(emojis.items()):
        if streak >= threshold:
            icon = emoji
    return f"{icon} *{streak}-day logging streak!* Keep it up!"


# ── COMMAND HANDLERS ──────────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = update.effective_user.first_name or "there"
    text = (
        f"👋 Hey {name}! I'm *CaLoRAify* — your AI nutrition assistant.\n\n"
        "📸 *Just send me a photo of any meal* and I'll instantly tell you:\n"
        "  • Ingredients detected\n"
        "  • Portion estimate\n"
        "  • Full calorie & macro breakdown\n\n"
        "📋 *Commands:*\n"
        "/today — today's calorie summary\n"
        "/week  — 7-day history chart\n"
        "/streak — your logging streak\n"
        "/goal [calories] — set daily calorie goal\n"
        "/delete [meal_id] — remove a logged meal\n"
        "/help — show this message"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, context)


async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id  = update.effective_user.id
    username = update.effective_user.username or ""
    today    = date.today().strftime("%Y-%m-%d")
    goal     = get_user_goal(user_id)
    summary  = get_daily_summary(user_id, today)

    if summary["meal_count"] == 0:
        await update.message.reply_text(
            "📭 No meals logged today yet.\n\nSend me a food photo to get started!",
            parse_mode="Markdown",
        )
        return

    await update.message.reply_text(
        format_daily_summary(summary, goal), parse_mode="Markdown"
    )


async def cmd_week(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    goal    = get_user_goal(user_id)
    data    = get_weekly_summary(user_id)
    await update.message.reply_text(
        format_weekly_chart(data, goal), parse_mode="Markdown"
    )


async def cmd_streak(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    streak  = get_streak(user_id)
    await update.message.reply_text(streak_message(streak), parse_mode="Markdown")


async def cmd_goal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id  = update.effective_user.id
    username = update.effective_user.username or ""

    if not context.args:
        goal = get_user_goal(user_id)
        await update.message.reply_text(
            f"🎯 Your current daily goal: *{goal} kcal*\n\n"
            "To change it: `/goal 1800`", parse_mode="Markdown"
        )
        return

    try:
        new_goal = int(context.args[0])
        if not (500 <= new_goal <= 10000):
            raise ValueError
    except ValueError:
        await update.message.reply_text(
            "❌ Please enter a number between 500 and 10000.\n"
            "Example: `/goal 1800`", parse_mode="Markdown"
        )
        return

    set_user_goal(user_id, username, new_goal)
    await update.message.reply_text(
        f"✅ Daily calorie goal updated to *{new_goal} kcal*!", parse_mode="Markdown"
    )


async def cmd_delete(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if not context.args:
        await update.message.reply_text(
            "Usage: `/delete <meal_id>`\n"
            "Find the meal ID in your /today summary.", parse_mode="Markdown"
        )
        return

    try:
        meal_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("❌ Invalid meal ID.", parse_mode="Markdown")
        return

    if delete_meal(user_id, meal_id):
        await update.message.reply_text(
            f"🗑 Meal #{meal_id} deleted.", parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(
            f"❌ Meal #{meal_id} not found or doesn't belong to you.",
            parse_mode="Markdown",
        )


# ── PHOTO HANDLER ─────────────────────────────────────────────────────────────
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id  = update.effective_user.id
    username = update.effective_user.username or str(user_id)

    # Acknowledge immediately so user knows we're working
    thinking_msg = await update.message.reply_text(
        "🔍 Analysing your meal… this takes 10–30 seconds ⏳"
    )

    try:
        # Download the highest-resolution version of the photo
        photo = update.message.photo[-1]
        file  = await context.bot.get_file(photo.file_id)
        buf   = io.BytesIO()
        await file.download_to_memory(buf)
        image_bytes = buf.getvalue()

        # Call the HF Space inference endpoint
        data = await analyze_food_image(image_bytes)

        # Log to database
        meal_id = log_meal(user_id, username, data)

        # Build response message
        reply = format_analysis_message(data, meal_id)

        # Add today's running total to the bottom
        today   = date.today().strftime("%Y-%m-%d")
        goal    = get_user_goal(user_id)
        summary = get_daily_summary(user_id, today)
        total   = summary["total_cal"]
        bar     = _bar(total, goal)
        reply  += f"\n\n📊 *Today so far:* {total:.0f} / {goal} kcal\n{bar}"

        # Check streak milestone
        streak = get_streak(user_id)
        if streak in (3, 7, 14, 30):
            reply += f"\n\n🏅 {streak_message(streak)}"

        # Delete the "analysing…" message and send the real reply
        await thinking_msg.delete()

        # Inline keyboard for quick actions
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📅 Today's summary", callback_data="today"),
                InlineKeyboardButton("📊 Week chart",      callback_data="week"),
            ],
            [
                InlineKeyboardButton(f"🗑 Delete meal #{meal_id}", callback_data=f"del:{meal_id}"),
            ],
        ])
        await update.message.reply_text(reply, parse_mode="Markdown", reply_markup=keyboard)

    except httpx.HTTPStatusError as e:
        await thinking_msg.delete()
        logger.error(f"API error: {e.response.status_code} {e.response.text}")
        await update.message.reply_text(
            "⚠️ The analysis service returned an error. "
            "Please try again in a moment.\n\n"
            "_If this keeps happening, the HuggingFace Space may be waking up — "
            "wait 60 seconds and resend._",
            parse_mode="Markdown",
        )
    except httpx.TimeoutException:
        await thinking_msg.delete()
        await update.message.reply_text(
            "⏱ Analysis timed out. The Space might be cold-starting — "
            "please try again in 30 seconds.",
            parse_mode="Markdown",
        )
    except Exception as e:
        await thinking_msg.delete()
        logger.exception(f"Unexpected error for user {user_id}: {e}")
        await update.message.reply_text(
            "❌ Something went wrong. Please try again.",
            parse_mode="Markdown",
        )


# ── CALLBACK QUERY HANDLER ────────────────────────────────────────────────────
async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query   = update.callback_query
    user_id = query.from_user.id
    data    = query.data
    await query.answer()

    if data == "today":
        today   = date.today().strftime("%Y-%m-%d")
        goal    = get_user_goal(user_id)
        summary = get_daily_summary(user_id, today)
        if summary["meal_count"] == 0:
            await query.message.reply_text("📭 No meals logged today yet.")
        else:
            await query.message.reply_text(
                format_daily_summary(summary, goal), parse_mode="Markdown"
            )

    elif data == "week":
        goal = get_user_goal(user_id)
        data_week = get_weekly_summary(user_id)
        await query.message.reply_text(
            format_weekly_chart(data_week, goal), parse_mode="Markdown"
        )

    elif data.startswith("del:"):
        meal_id = int(data.split(":")[1])
        if delete_meal(user_id, meal_id):
            await query.message.reply_text(f"🗑 Meal #{meal_id} removed.")
        else:
            await query.message.reply_text(f"❌ Couldn't delete meal #{meal_id}.")

    elif data == "streak":
        streak = get_streak(user_id)
        await query.message.reply_text(streak_message(streak), parse_mode="Markdown")


# ── TEXT FALLBACK ─────────────────────────────────────────────────────────────
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📸 Send me a photo of your meal and I'll analyse it!\n"
        "Use /help to see all commands.",
        parse_mode="Markdown",
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        raise RuntimeError(
            "Set TELEGRAM_BOT_TOKEN environment variable before starting the bot.\n"
            "Get a token from @BotFather on Telegram."
        )
    if "YOUR_SPACE_URL" in API_URL:
        raise RuntimeError(
            "Set CALORAIFY_API_URL to your HuggingFace Space URL.\n"
            "Example: https://yourname-caloraify.hf.space"
        )

    init_db()
    logger.info(f"Database initialised at {DB_PATH}")
    logger.info(f"Connecting to API at {API_URL}")

    app = Application.builder().token(BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("help",   cmd_help))
    app.add_handler(CommandHandler("today",  cmd_today))
    app.add_handler(CommandHandler("week",   cmd_week))
    app.add_handler(CommandHandler("streak", cmd_streak))
    app.add_handler(CommandHandler("goal",   cmd_goal))
    app.add_handler(CommandHandler("delete", cmd_delete))

    # Messages
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Inline buttons
    app.add_handler(CallbackQueryHandler(handle_callback))

    logger.info("CaLoRAify bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
