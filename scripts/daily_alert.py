"""
Daily alert check — runs as part of the GitHub Actions workflow.

Generates the intelligence brief, checks for headline-level (importance=1)
observations, and outputs a structured alert suitable for webhook delivery
(Discord, Slack, email, etc).

Exit codes:
  0 — ran successfully, alert written to stdout / file
  1 — error (no data, broken dependency)

Usage:
    python -m scripts.daily_alert                    # print alert to stdout
    python -m scripts.daily_alert --json             # JSON format
    python -m scripts.daily_alert --webhook URL      # POST to webhook
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.advisory.buying_brief import generate_brief, _generate_headline


def build_alert() -> dict:
    """Generate the intelligence brief and extract alert-worthy findings."""
    brief = generate_brief("coffee")
    headline = _generate_headline(brief.observations)

    # Separate by importance
    urgent = [o for o in brief.observations if o.importance == 1]
    notable = [o for o in brief.observations if o.importance == 2]
    background = [o for o in brief.observations if o.importance == 3]

    # Determine alert level
    if urgent:
        level = "high"
    elif notable:
        level = "normal"
    else:
        level = "quiet"

    alert = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "as_of_date": brief.as_of_date,
        "price": brief.current_price,
        "level": level,
        "headline": headline,
        "urgent_count": len(urgent),
        "notable_count": len(notable),
        "urgent": [
            {"topic": o.topic, "text": o.text, "source": o.source}
            for o in urgent
        ],
        "notable": [
            {"topic": o.topic, "text": o.text, "source": o.source}
            for o in notable
        ],
    }

    return alert


def format_discord(alert: dict) -> dict:
    """Format alert as a Discord webhook payload."""
    level_emoji = {"high": ":red_circle:", "normal": ":yellow_circle:", "quiet": ":green_circle:"}
    level_color = {"high": 0xC0392B, "normal": 0xF39C12, "quiet": 0x27AE60}

    embed = {
        "title": f"{level_emoji.get(alert['level'], '')} Coffee Market Alert",
        "description": alert["headline"],
        "color": level_color.get(alert["level"], 0x95A5A6),
        "fields": [],
        "footer": {"text": f"Price: {alert['price']:.2f} c/lb · Data: {alert['as_of_date']}"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if alert["urgent"]:
        for item in alert["urgent"][:3]:
            embed["fields"].append({
                "name": f":rotating_light: {item['topic']}",
                "value": item["text"][:1024],
                "inline": False,
            })

    if alert["notable"]:
        notable_text = "\n".join(
            f"• **{item['topic']}**: {item['text'][:200]}"
            for item in alert["notable"][:5]
        )
        embed["fields"].append({
            "name": "Notable",
            "value": notable_text[:1024],
            "inline": False,
        })

    return {"embeds": [embed]}


def format_slack(alert: dict) -> dict:
    """Format alert as a Slack incoming webhook payload."""
    level_emoji = {"high": ":red_circle:", "normal": ":large_yellow_circle:", "quiet": ":large_green_circle:"}

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"Coffee Market Alert — {alert['as_of_date']}"}
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn",
                     "text": f"{level_emoji.get(alert['level'], '')} *{alert['headline']}*"}
        },
    ]

    if alert["urgent"]:
        for item in alert["urgent"][:3]:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn",
                         "text": f":rotating_light: *{item['topic']}*\n{item['text'][:500]}"}
            })

    if alert["notable"]:
        notable_text = "\n".join(
            f"• *{item['topic']}*: {item['text'][:200]}"
            for item in alert["notable"][:5]
        )
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": notable_text[:3000]}
        })

    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn",
                       "text": f"Price: {alert['price']:.2f} c/lb · Research tool, not financial advice"}]
    })

    return {"blocks": blocks}


def send_webhook(url: str, alert: dict) -> bool:
    """POST alert to a webhook URL (Discord or Slack format auto-detected)."""
    import urllib.request
    import urllib.error

    if "discord" in url.lower():
        payload = format_discord(alert)
    else:
        payload = format_slack(alert)

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status < 300
    except urllib.error.URLError as e:
        print(f"Webhook failed: {e}", file=sys.stderr)
        return False


def format_text(alert: dict) -> str:
    """Plain text format for stdout / GitHub step summary."""
    lines = [
        f"COFFEE MARKET ALERT — {alert['as_of_date']}",
        f"Level: {alert['level'].upper()}",
        f"Price: {alert['price']:.2f} cents/lb",
        "",
        alert["headline"],
        "",
    ]

    if alert["urgent"]:
        lines.append("URGENT:")
        for item in alert["urgent"]:
            lines.append(f"  [{item['topic']}] {item['text']}")
            lines.append(f"    Source: {item['source']}")
        lines.append("")

    if alert["notable"]:
        lines.append("NOTABLE:")
        for item in alert["notable"]:
            lines.append(f"  [{item['topic']}] {item['text']}")
        lines.append("")

    lines.append("---")
    lines.append("Research tool. Not financial advice.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Daily coffee market alert check.")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--webhook", type=str, help="POST alert to this webhook URL")
    parser.add_argument("--quiet-skip", action="store_true",
                        help="Exit silently (no output) when level is 'quiet'")
    args = parser.parse_args()

    try:
        alert = build_alert()
    except Exception as e:
        print(f"Failed to generate alert: {e}", file=sys.stderr)
        sys.exit(1)

    # Skip if nothing noteworthy and --quiet-skip is set
    if args.quiet_skip and alert["level"] == "quiet":
        sys.exit(0)

    if args.json:
        print(json.dumps(alert, indent=2, default=str))
    else:
        print(format_text(alert))

    if args.webhook:
        ok = send_webhook(args.webhook, alert)
        if not ok:
            print("Webhook delivery failed.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
