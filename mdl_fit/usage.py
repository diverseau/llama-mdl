"""What the machine holds when nothing of yours is open.

A scan taken mid-game, or with a browser full of tabs, sees a card and a
RAM that are mostly taken. What a config has to fit is the machine as it
is when you sit down to run a model: the OS, the desktop, and whatever
starts with it. That figure is a mix of three things, best first:

  set/seen  the machine at idle - set with `mdl fit hw --idle`, or seen
            by any probe in the first minutes after boot
  measured  what is in use now, less what the apps opened since boot
            hold. An app is anything that is neither the system's, nor
            a startup entry's, nor the terminal this runs in
  typical   what the OS and its desktop need at the least; a floor under
            the measured figure, and all there is when nothing else is

Totals come from the adapter and the OS, never from summing processes:
per-process VRAM counters double-count (Windows books shared surfaces to
dwm, which can read 9 G on a card using 1.5). Per-process figures are
only used to find what to take back off the total.
"""

import csv
import ctypes
import os
import plistlib
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

MiB = 1 << 20
GiB = 1 << 30
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
BOOT_WINDOW = 10 * 60      # a probe this soon after boot sees the idle box
SAMPLES = 10

# The OS and its desktop with nothing else open, (RAM, VRAM). Rough
# figures for a clean install at 1080p-1440p, erring low: this is a
# floor, not a forecast.
TYPICAL = {
    "windows11": (3 * GiB, 300 * MiB),
    "windows10": (int(2.5 * GiB), 250 * MiB),
    "macos": (int(3.5 * GiB), 0),        # unified memory: no separate card
    "gnome": (int(1.5 * GiB), 250 * MiB),
    "kde": (int(1.2 * GiB), 200 * MiB),
    "light": (600 * MiB, 100 * MiB),     # xfce, lxqt, mate, a tiling wm
    "linux": (int(1.2 * GiB), 200 * MiB),  # a desktop this cannot name
    "headless": (400 * MiB, 0),
}
# What a browser holds is what you opened in it, whatever started it.
BROWSERS = {"chrome", "msedge", "firefox", "brave", "opera", "vivaldi",
            "comet", "arc", "safari", "chromium", "zen", "librewolf"}
# Desktop parts that run as the user on Linux and macOS.
DESKTOP = ("gnome-", "gsd-", "kwin", "plasmashell", "kded", "ksmserver",
           "kactivitymanagerd", "xorg", "xwayland", "pipewire", "wireplumber",
           "pulseaudio", "dbus", "systemd", "at-spi", "gvfs", "xdg-", "ibus",
           "fcitx", "xfce4-", "xfwm4", "xfdesktop", "lxqt-", "sway",
           "hyprland", "waybar", "mutter", "evolution-", "tracker-", "goa-",
           "polkit", "dock", "finder", "systemuiserver", "controlcenter",
           "windowmanager", "loginwindow", "notificationcenter", "spotlight")


def _run(argv, timeout=10):
    try:
        p = subprocess.run(argv, capture_output=True, text=True,
                           timeout=timeout, creationflags=NO_WINDOW,
                           errors="replace")
        return p.stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def norm(name):
    """'C:\\...\\Discord.exe' -> 'discord'."""
    base = str(name).replace("\\", "/").rsplit("/", 1)[-1].lower()
    return re.sub(r"\.(exe|app)$", "", base)


def os_key(platform=None, env=None, winbuild=None):
    platform = platform or sys.platform
    env = os.environ if env is None else env
    if platform.startswith("win"):
        if winbuild is None:
            winbuild = getattr(sys, "getwindowsversion",
                               lambda: type("v", (), {"build": 22000}))().build
        return "windows11" if winbuild >= 22000 else "windows10"
    if platform == "darwin":
        return "macos"
    if not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")):
        return "headless"
    de = (env.get("XDG_CURRENT_DESKTOP") or env.get("DESKTOP_SESSION")
          or "").lower()
    for key, names in (("gnome", ("gnome", "ubuntu", "unity", "cinnamon",
                                  "budgie", "pantheon")),
                       ("kde", ("kde", "plasma")),
                       ("light", ("xfce", "lxqt", "lxde", "mate", "sway",
                                  "hyprland", "i3", "niri"))):
        if any(n in de for n in names):
            return key
    return "linux"


# ------------------------------------------------------------ startup --

def exe_of(cmd):
    """The program a startup command runs, as a process name. Squirrel
    installers (Discord, Medal) start Update.exe --processStart X.exe."""
    m = re.search(r'--processStart\s+"?([^"\s]+)', cmd)
    if m:
        return norm(m.group(1))
    m = (re.match(r'\s*"([^"]+)"', cmd)
         or re.match(r"\s*(.+?\.exe)\b", cmd, re.I)
         or re.match(r"\s*(\S+)", cmd))
    return norm(m.group(1)) if m else ""


def enabled(approved):
    """Task Manager's StartupApproved bytes: an odd first byte is off."""
    return not approved or approved[0] % 2 == 0


def desktop_exec(text):
    """The program an XDG autostart .desktop file runs, or None when the
    entry is switched off."""
    fields = {}
    for line in text.splitlines():
        if line.startswith("[") and fields:
            break                          # only the [Desktop Entry] group
        k, sep, v = line.partition("=")
        if sep:
            fields.setdefault(k.strip(), v.strip())
    if fields.get("Hidden", "").lower() == "true" or fields.get(
            "X-GNOME-Autostart-enabled", "").lower() == "false":
        return None
    words = [w for w in fields.get("Exec", "").split()
             if not w.startswith("%") and "=" not in w and w != "env"]
    if len(words) >= 3 and norm(words[0]) == "flatpak" and words[1] == "run":
        app = [w for w in words[2:] if not w.startswith("-")]
        if not app:
            return None
        # com.discordapp.Discord -> discord, com.spotify.Client -> spotify
        parts = app[0].lower().split(".")
        generic = {"client", "desktop", "app", "application"}
        return parts[-2] if parts[-1] in generic and len(parts) > 1 \
            else parts[-1]
    return norm(words[0]) if words else None


def startup_names():
    """Process names the OS starts at login, lower case, without .exe."""
    try:
        if os.name == "nt":
            names = _startup_windows()
        elif sys.platform == "darwin":
            names = _startup_mac()
        else:
            names = _startup_linux()
    except (OSError, ValueError, ImportError):
        return set()
    names.discard("")
    names.discard(None)
    return names


def _reg_values(root, path):
    import winreg
    out = {}
    try:
        with winreg.OpenKey(root, path) as key:
            i = 0
            while True:
                try:
                    name, value, _ = winreg.EnumValue(key, i)
                except OSError:
                    break
                out[name] = value
                i += 1
    except OSError:
        pass
    return out


def _startup_windows():
    import winreg
    run = r"Software\Microsoft\Windows\CurrentVersion\Run"
    approved = r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\%s"
    names, folder_ok = set(), {}
    for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        ok = _reg_values(root, approved % "Run")
        ok.update(_reg_values(root, approved % "Run32"))
        folder_ok.update(_reg_values(root, approved % "StartupFolder"))
        for path in (run, run.replace("Software\\", "Software\\WOW6432Node\\")):
            for name, cmd in _reg_values(root, path).items():
                if isinstance(cmd, str) and enabled(ok.get(name)):
                    names.add(exe_of(cmd))
    dirs = [Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/"
            "Programs/Startup",
            Path(os.environ.get("ProgramData", "")) / "Microsoft/Windows/"
            "Start Menu/Programs/StartUp"]
    for d in dirs:
        for f in d.glob("*") if d.is_dir() else []:
            if f.suffix.lower() in (".lnk", ".exe", ".cmd", ".bat", ".url") \
                    and enabled(folder_ok.get(f.name)):
                names.add(norm(f.stem))
    return names


def _startup_linux():
    names, seen = set(), set()
    home = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    for d in [home / "autostart"] + [Path(p) / "autostart" for p in (
            os.environ.get("XDG_CONFIG_DIRS") or "/etc/xdg").split(":")]:
        for f in sorted(d.glob("*.desktop")) if d.is_dir() else []:
            if f.name in seen:             # the user's copy overrides
                continue
            seen.add(f.name)
            try:
                name = desktop_exec(f.read_text(errors="replace"))
            except OSError:
                continue
            if name:
                names.add(name)
    return names


def _startup_mac():
    names = set()
    for d in (Path.home() / "Library/LaunchAgents",
              Path("/Library/LaunchAgents")):
        for f in d.glob("*.plist") if d.is_dir() else []:
            try:
                with f.open("rb") as fh:
                    plist = plistlib.load(fh)
            except (OSError, ValueError, plistlib.InvalidFileException):
                continue
            if plist.get("Disabled"):
                continue
            prog = plist.get("Program") or (plist.get("ProgramArguments")
                                            or [""])[0]
            names.add(norm(prog))
    return names


def is_startup(name, startup):
    """steam starts steamwebhelper; a name that begins with a startup
    entry's is taken to be one of its own."""
    return name in startup or any(len(s) >= 4 and name.startswith(s)
                                  for s in startup)


# ---------------------------------------------------------- processes --

class Proc:
    __slots__ = ("pid", "ppid", "name", "path", "ram", "vram", "system",
                 "kind")

    def __init__(self, pid, ppid, name, path="", ram=0, vram=0,
                 system=False):
        self.pid, self.ppid, self.name, self.path = pid, ppid, name, path
        self.ram, self.vram, self.system = ram, vram, system
        self.kind = None

    def __repr__(self):
        return "Proc(%d %s %s)" % (self.pid, self.name, self.kind)


def classify(procs, startup, here=None):
    """Tag every process 'system', 'startup', 'here' (the terminal this
    runs in, which is still open when the model loads) or 'app'.
    A child takes its parent's tag, unless the parent is the system or
    the terminal: whatever explorer or a shell starts is an app."""
    by = {p.pid: p for p in procs}
    chain, pid = set(), os.getpid() if here is None else here
    for _ in range(64):
        if pid in chain or pid not in by:
            break
        chain.add(pid)
        pid = by[pid].ppid

    def own(p):
        if p.system:
            return "system"
        if p.pid in chain:
            return "here"
        if p.name in BROWSERS:
            return "app"
        if is_startup(p.name, startup):
            return "startup"
        return None

    def kind(p, depth=0):
        if p.kind:
            return p.kind
        k = own(p)
        if k is None:
            parent = by.get(p.ppid)
            if parent is None or parent is p or depth > 32:
                k = "app"
            else:
                pk = kind(parent, depth + 1)
                k = "app" if pk in ("system", "here") else pk
        p.kind = k
        return k

    for p in procs:
        kind(p)
    return procs


def processes(quick=False):
    """[Proc] with RAM (and, where the OS will say, VRAM) per process;
    (procs, whether the VRAM figures can be trusted to be there)."""
    try:
        if os.name == "nt":
            procs = _procs_windows()
            vram = {} if quick else parse_typeperf(_run(
                ["typeperf", r"\GPU Process Memory(*)\Dedicated Usage",
                 "-sc", "1"], timeout=15))
            for p in procs:
                p.vram = vram.get(p.pid, 0)
            return procs, bool(vram)
        if sys.platform == "darwin":
            return _procs_mac(), False
        procs = _procs_linux(quick)
        have = any(p.vram for p in procs)
        if not quick and shutil.which("nvidia-smi"):
            got, ok = parse_nvidia_procs(_run(["nvidia-smi"]))
            for p in procs:
                p.vram += got.get(p.pid, 0)
            have = have or ok
        return procs, have
    except (OSError, ValueError, AttributeError):
        return [], False


def parse_typeperf(text):
    """{pid: bytes} from typeperf's CSV of GPU Process Memory counters."""
    rows = list(csv.reader(line for line in text.splitlines()
                           if line.startswith('"')))
    if len(rows) < 2:
        return {}
    out = {}
    for head, value in zip(rows[0][1:], rows[1][1:], strict=False):
        m = re.search(r"pid_(\d+)_", head)
        try:
            n = int(float(value))
        except ValueError:
            continue
        if m and n > 0:
            pid = int(m.group(1))
            out[pid] = out.get(pid, 0) + n
    return out


_NV_PROC = re.compile(r"^\|\s+\d+\s+\S+\s+\S+\s+(\d+)\s+(?:C\+G|C|G)\s+.*?"
                      r"(\d+)MiB\s*\|", re.M)


def parse_nvidia_procs(text):
    """({pid: bytes}, ok) from nvidia-smi's process table. On Windows the
    column reads N/A, which is why Windows uses the GPU counters."""
    ok = "Processes:" in text
    return ({int(m.group(1)): int(m.group(2)) * MiB
             for m in _NV_PROC.finditer(text)}, ok)


def _procs_windows():
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    dword, handle = ctypes.c_ulong, ctypes.c_void_p

    class Entry(ctypes.Structure):
        _fields_ = [("size", dword), ("usage", dword), ("pid", dword),
                    ("heap", ctypes.c_size_t), ("module", dword),
                    ("threads", dword), ("ppid", dword),
                    ("pri", ctypes.c_long), ("flags", dword),
                    ("exe", ctypes.c_wchar * 260)]

    class Counters(ctypes.Structure):
        _fields_ = [("cb", dword), ("faults", dword)] + [
            (n, ctypes.c_size_t) for n in (
                "peak_ws", "ws", "peak_paged", "paged", "peak_nonpaged",
                "nonpaged", "pagefile", "peak_pagefile")]

    k32.CreateToolhelp32Snapshot.restype = handle
    k32.CreateToolhelp32Snapshot.argtypes = [dword, dword]
    k32.Process32FirstW.argtypes = [handle, ctypes.POINTER(Entry)]
    k32.Process32NextW.argtypes = [handle, ctypes.POINTER(Entry)]
    k32.OpenProcess.restype = handle
    k32.OpenProcess.argtypes = [dword, ctypes.c_int, dword]
    k32.K32GetProcessMemoryInfo.argtypes = [handle, ctypes.POINTER(Counters),
                                            dword]
    k32.QueryFullProcessImageNameW.argtypes = [handle, dword,
                                               ctypes.c_wchar_p,
                                               ctypes.POINTER(dword)]
    k32.ProcessIdToSessionId.argtypes = [dword, ctypes.POINTER(dword)]
    k32.CloseHandle.argtypes = [handle]
    snap = k32.CreateToolhelp32Snapshot(2, 0)     # TH32CS_SNAPPROCESS
    if not snap or snap == ctypes.c_void_p(-1).value:
        return []
    windir = os.environ.get("SystemRoot", r"C:\Windows").lower()
    out = []
    try:
        e = Entry()
        e.size = ctypes.sizeof(Entry)
        more = k32.Process32FirstW(snap, ctypes.byref(e))
        while more:
            p = Proc(e.pid, e.ppid, norm(e.exe))
            sid = dword()
            session = k32.ProcessIdToSessionId(e.pid, ctypes.byref(sid))
            h = k32.OpenProcess(0x1000, False, e.pid)   # QUERY_LIMITED
            if h:
                try:
                    c = Counters()
                    c.cb = ctypes.sizeof(Counters)
                    if k32.K32GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
                        p.ram = c.ws
                    buf = ctypes.create_unicode_buffer(1024)
                    n = dword(1024)
                    if k32.QueryFullProcessImageNameW(h, 0, buf,
                                                      ctypes.byref(n)):
                        p.path = buf.value
                finally:
                    k32.CloseHandle(h)
            # services (session 0), anything under Windows, and anything
            # too protected to open belong to the system
            p.system = (e.pid in (0, 4) or not session or sid.value == 0
                        or not p.path or p.path.lower().startswith(windir))
            out.append(p)
            more = k32.Process32NextW(snap, ctypes.byref(e))
    finally:
        k32.CloseHandle(snap)
    return out


def _procs_linux(quick):
    me, out = os.getuid(), []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            st = {}
            for line in (d / "status").read_text().splitlines():
                k, _, v = line.partition(":")
                st[k] = v.split()
        except OSError:
            continue
        name = norm(" ".join(st.get("Name", [""])))
        p = Proc(int(d.name), int((st.get("PPid") or ["0"])[0]), name)
        p.ram = int((st.get("VmRSS") or ["0"])[0]) * 1024
        uid = int((st.get("Uid") or ["0"])[0])
        try:
            p.path = os.readlink(d / "exe")
        except OSError:
            pass
        p.system = uid != me or name.startswith(DESKTOP)
        if not quick and uid == me:
            p.vram = _drm_vram(d)
        out.append(p)
    return out


def _drm_vram(d):
    """VRAM a process holds through /dev/dri, from fdinfo (amdgpu, i915,
    xe, nouveau). NVIDIA's driver does not fill these in."""
    total, clients = 0, set()
    try:
        fds = list((d / "fd").iterdir())
    except OSError:
        return 0
    for fd in fds:
        try:
            if not os.readlink(fd).startswith("/dev/dri/"):
                continue
            text = (d / "fdinfo" / fd.name).read_text()
        except OSError:
            continue
        client = re.search(r"drm-client-id:\s*(\d+)", text)
        if client:
            if client.group(1) in clients:
                continue
            clients.add(client.group(1))
        for m in re.finditer(r"drm-(?:memory|total)-vram\d*:\s*(\d+)\s*(KiB|MiB)?",
                             text):
            total += int(m.group(1)) * (MiB if m.group(2) == "MiB" else 1024)
    return total


def _procs_mac():
    me, out = os.getuid(), []
    for line in _run(["ps", "-axo", "pid=,ppid=,uid=,rss=,comm="]).splitlines():
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        pid, ppid, uid, rss, comm = parts
        p = Proc(int(pid), int(ppid), norm(comm), comm, int(rss) * 1024)
        p.system = (int(uid) != me or p.name.startswith(DESKTOP)
                    or comm.startswith(("/System/", "/usr/", "/sbin/",
                                        "/Library/Apple/")))
        out.append(p)
    return out


# ------------------------------------------------------- load and time --

def cpu_load(interval=0.25):
    """Fraction of the CPU busy over `interval`, or None."""
    try:
        if os.name == "nt":
            k32 = ctypes.WinDLL("kernel32")

            def sample():
                i, k, u = (ctypes.c_ulonglong() for _ in range(3))
                k32.GetSystemTimes(ctypes.byref(i), ctypes.byref(k),
                                   ctypes.byref(u))
                return i.value, k.value + u.value   # kernel time has idle in it
        elif Path("/proc/stat").is_file():
            def sample():
                f = [int(x) for x in
                     Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]]
                return f[3] + (f[4] if len(f) > 4 else 0), sum(f)
        else:
            return None
        i0, t0 = sample()
        time.sleep(interval)
        i1, t1 = sample()
        return max(0.0, min(1.0, 1 - (i1 - i0) / max(1, t1 - t0)))
    except (OSError, ValueError, AttributeError):
        return None


def uptime():
    """Seconds since boot, or None."""
    try:
        if os.name == "nt":
            k32 = ctypes.WinDLL("kernel32")
            k32.GetTickCount64.restype = ctypes.c_ulonglong
            return k32.GetTickCount64() / 1000.0
        if Path("/proc/uptime").is_file():
            return float(Path("/proc/uptime").read_text().split()[0])
        m = re.search(r"sec = (\d+)", _run(["sysctl", "-n", "kern.boottime"]))
        return time.time() - int(m.group(1)) if m else None
    except (OSError, ValueError, AttributeError):
        return None


# ------------------------------------------------------------ snapshot --

class Snapshot:
    """The machine this minute: totals in use, and who holds what."""

    def __init__(self, key, procs=(), have_vram=False, vram_used=None,
                 ram_used=None, cpu=None, up=None):
        self.os_key, self.procs, self.have_vram = key, list(procs), have_vram
        self.vram_used, self.ram_used = vram_used, ram_used
        self.cpu, self.uptime = cpu, up

    @property
    def apps(self):
        return [p for p in self.procs if p.kind == "app"]

    def held(self, what):
        """[(name, bytes)] of what open apps hold, biggest first."""
        by = {}
        for p in self.apps:
            by[p.name] = by.get(p.name, 0) + getattr(p, what)
        return sorted(((n, b) for n, b in by.items() if b >= 32 * MiB),
                      key=lambda x: -x[1])


def snapshot(vram_used=None, ram_used=None, quick=False):
    procs, have = processes(quick)
    classify(procs, startup_names())
    return Snapshot(os_key(), procs, have, vram_used, ram_used,
                    None if quick else cpu_load(), uptime())


class Baseline:
    """What the OS and resident programs hold, and how that was known:
    'set', 'seen at boot', 'measured', 'now' or 'typical'."""

    def __init__(self):
        self.vram = self.ram = 0
        self.vram_how = self.ram_how = "typical"
        self.measured_vram = self.measured_ram = None


def learned(saved, what):
    """(bytes, how) from hw.json's idle record, or (None, None)."""
    fixed = (saved or {}).get("set", {})
    if fixed.get(what) is not None:
        return int(fixed[what]), "set"
    seen = sorted(s[what] for s in (saved or {}).get("seen", [])
                  if s.get(what) is not None)
    if seen:
        return int(seen[len(seen) // 2]), "seen at boot"
    return None, None


def baseline(snap, saved=None):
    typ_ram, typ_vram = TYPICAL.get(snap.os_key, TYPICAL["linux"])
    b, apps = Baseline(), snap.apps
    if snap.ram_used is not None:
        b.measured_ram = max(typ_ram, snap.ram_used - sum(p.ram for p in apps))
    if snap.vram_used is not None and snap.have_vram:
        b.measured_vram = max(typ_vram,
                              snap.vram_used - sum(p.vram for p in apps))
    for what, typ, meas, now in (
            ("ram", typ_ram, b.measured_ram, snap.ram_used),
            ("vram", typ_vram, b.measured_vram, snap.vram_used)):
        val, how = learned(saved, what)
        if val is None:
            val, how = ((meas, "measured") if meas is not None else
                        (now, "now") if now is not None else (typ, "typical"))
        setattr(b, what, int(val))
        setattr(b, what + "_how", how)
    return b


def record_boot(saved, snap, b):
    """Book this probe as an idle sample if it came soon after boot.
    One sample per boot; the last SAMPLES are kept. True if booked."""
    if snap.uptime is None or snap.uptime > BOOT_WINDOW:
        return False
    boot = int(time.time() - snap.uptime) // 600
    seen = saved.setdefault("seen", [])
    if any(s.get("boot") == boot for s in seen):
        return False
    seen.append({"boot": boot, "ram": b.measured_ram, "vram": b.measured_vram,
                 "at": time.strftime("%Y-%m-%d %H:%M")})
    del seen[:-SAMPLES]
    return True


def plan_free(total, free_now, used_now, base):
    """Free VRAM with the apps opened since boot closed. Backends count
    'free' their own way (Vulkan's budget is more generous than total
    less used), so what closing apps gives back is added to their figure
    and capped at the card less the baseline."""
    if used_now is None or base is None or not total:
        return free_now
    back = max(0, used_now - base)
    return max(free_now, min(total - base, free_now + back))
