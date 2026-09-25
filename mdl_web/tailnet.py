"""A running model on your tailnet, through `tailscale serve`.

Only a server with an --api-key is shared: the tailnet is every machine
on it, and a server without a key answers anyone who reaches it. The
share is HTTPS on the server's own port, at this machine's tailnet name.
"""

import json
import shutil
import subprocess


def _run(argv, timeout=20):
    exe = shutil.which("tailscale")
    if not exe:
        raise OSError("tailscale is not installed")
    return subprocess.run([exe] + argv, capture_output=True, text=True,
                          timeout=timeout,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW",
                                                0))


def available():
    return bool(shutil.which("tailscale"))


def name():
    """This machine's name on the tailnet, or None when it is not on one."""
    try:
        p = _run(["status", "--json"], timeout=8)
        me = json.loads(p.stdout or "{}").get("Self") or {}
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    dns = (me.get("DNSName") or "").rstrip(".")
    return dns or None


def share(port):
    """Serve 127.0.0.1:port on the tailnet; the URL, or raise OSError."""
    host = name()
    if not host:
        raise OSError("this machine is not on a tailnet (tailscale up)")
    p = _run(["serve", "--bg", "--https=%d" % port,
              "http://127.0.0.1:%d" % port])
    if p.returncode != 0:
        raise OSError((p.stderr or p.stdout or "tailscale serve failed")
                      .strip().splitlines()[-1])
    return "https://%s:%d" % (host, port)


def unshare(port):
    try:
        _run(["serve", "--https=%d" % port, "off"])
    except (OSError, subprocess.SubprocessError):
        pass
