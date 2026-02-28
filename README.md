# Profile Picture Validator

A Django service that validates profile pictures using OpenCV.
It checks that uploaded images contain a **real human face** — rejecting cartoons, bitmojis, anime art, rotated photos, placeholders, and non-face images.

## Running Locally

```bash
uv run python manage.py runserver
```

Or via `main.py`:

```bash
uv run python main.py
```

## Running in Production

```bash
pip install gunicorn
gunicorn pfp_validator.wsgi
```

## Environment Variables

| Variable               | Default               | Description                                 |
| ---------------------- | --------------------- | ------------------------------------------- |
| `DJANGO_SECRET_KEY`    | insecure dev key      | Set to a strong random string in production |
| `DJANGO_DEBUG`         | `False`               | Set to `True` only for local development    |
| `DJANGO_ALLOWED_HOSTS` | `127.0.0.1,localhost` | Comma-separated list of allowed hosts       |

## API

**POST** `/analyze/`

Form field: `image` (multipart, max 5 MB)

```json
{
  "valid": true,
  "score": 100,
  "messages": ["Great profile photo!"]
}
```

## Rejection Reasons

- No face detected
- Face too small / partial / cut off
- Image is sideways or upside-down (with specific rotation instruction)
- Illustration / anime / cartoon artwork
- Flat-coloured bitmoji or digital avatar
- Large object blocking the face
- Blank / placeholder image
