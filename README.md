
# 🧠 Deep Learning — Master 2 · Course Slides

Lecture slides for the **Deep Learning** graduate course, hosted on GitHub Pages with time-limited access control.

---

## 🗂️ File Structure

```
.
├── index.html          # Password-protected landing page
├── presentation.html   # Reveal.js lecture slides
├── style.css           # Shared dark theme stylesheet
└── README.md           # This file
```

---

## 🚀 Deployment (GitHub Pages)

1. Push all files to the **root** of your repository's `main` branch.
2. Go to **Settings → Pages**.
3. Under *Source*, select **Deploy from a branch** → `main` → `/ (root)`.
4. Click **Save**. Your site will be live at:
   ```
   https://<your-username>.github.io/<repo-name>/
   ```

---

## 🔑 Changing the Password

The password is stored as a **SHA-256 hash** in `index.html`.

### Step 1 — Generate the hash

Open your browser console (`F12`) and run:

```javascript
const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode("yourNewPassword"));
console.log(Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2,"0")).join(""));
```

Copy the output (64-character hex string).

### Step 2 — Update `index.html`

Find this line and replace the hash value:

```javascript
const PASSWORD_HASH = "cb4bbee11bf9cc7df5819358fa465b21aa17df946b7013f2b6a46cc016e78dad";
```

> **Default password:** `deeplearning2025`

---

## ⏰ Changing the Expiry Date

The expiry date is Base64-encoded in `index.html`.

### Step 1 — Encode your new date

In the browser console:

```javascript
btoa("2027-06-30")   // → "MjAyNy0wNi0zMA=="
```

### Step 2 — Update `index.html`

```javascript
const EXPIRY_B64 = "MjAyNy0wNi0zMA==";
```

> **Current expiry:** `2026-07-01`

---

## 🛡️ Security Notes

| Mechanism | What it protects against |
|---|---|
| SHA-256 password hash | Casual source-code inspection |
| Base64-encoded expiry | Trivial date tampering |
| 3-attempt limit + exponential lock | Manual brute-force |
| Artificial delay on submit | Timing-based enumeration |

⚠️ **This is client-side security.** A determined developer with DevTools can bypass it.  
It is appropriate for **internal / semi-private** use (e.g. enrolled students only).  
Do **not** use this to protect genuinely sensitive or confidential data.

---

## ✏️ Customising the Slides

`presentation.html` uses [Reveal.js 5](https://revealjs.com/) loaded from CDN.  
Each `<section>` tag is one slide. Vertical slides are nested `<section>` inside a `<section>`.

Useful Reveal.js options (bottom of `presentation.html`):

```javascript
Reveal.initialize({
  transition: "slide",   // none | fade | slide | convex | concave | zoom
  slideNumber: "c/t",    // current/total
  hash: true,            // URL updates with slide number
});
```

---

## 📄 License

Course material — all rights reserved. 
Redistribution outside enrolled students is not permitted.
