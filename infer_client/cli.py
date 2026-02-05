"""Interactive CLI: ``infer`` command — Rich UI with guided menus."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from . import credentials
from .client import InferClient

console = Console()

# Global client reference (set in main())
_client: Optional[InferClient] = None

# ──────────────────────────────────────────────
# Banner
# ──────────────────────────────────────────────

BANNER_COLOR = "bright_cyan"

BANNER_ASCII = (
    "██╗███╗   ██╗███████╗███████╗██████╗ \n"
    "██║████╗  ██║██╔════╝██╔════╝██╔══██╗\n"
    "██║██╔██╗ ██║█████╗  █████╗  ██████╔╝\n"
    "██║██║╚██╗██║██╔══╝  ██╔══╝  ██╔══██╗\n"
    "██║██║ ╚████║██║     ███████╗██║  ██║\n"
    "╚═╝╚═╝  ╚═══╝╚═╝     ╚══════╝╚═╝  ╚═╝\n"
    "\n"
    " ██████╗██╗     ██╗███████╗███╗   ██╗████████╗\n"
    "██╔════╝██║     ██║██╔════╝████╗  ██║╚══██╔══╝\n"
    "██║     ██║     ██║█████╗  ██╔██╗ ██║   ██║   \n"
    "██║     ██║     ██║██╔══╝  ██║╚██╗██║   ██║   \n"
    "╚██████╗███████╗██║███████╗██║ ╚████║   ██║   \n"
    " ╚═════╝╚══════╝╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝   "
)

BANNER_TAGLINE = "LLM Tool Inference Client — Classify & Generate at Scale"


def _print_banner() -> None:
    console.print()
    for line in BANNER_ASCII.split("\n"):
        console.print(Align.center(f"[bold {BANNER_COLOR}]{line}[/bold {BANNER_COLOR}]"))
    console.print()
    console.print(Align.center(f"[dim]{BANNER_TAGLINE}[/dim]"))
    console.print()


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _warn(msg: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {msg}")


def _separator() -> None:
    console.print()


def _press_enter() -> None:
    console.print()
    Prompt.ask("[dim]Press Enter to return to menu[/dim]", default="")


def _ask_model(client: InferClient) -> Optional[str]:
    """Interactively ask for a model, or return None for server default."""
    try:
        with console.status("[bold cyan]Fetching models...[/bold cyan]", spinner="dots"):
            models = client.models()
    except Exception as e:
        _warn(f"Could not fetch models: {e}")
        return None

    if not models:
        console.print("[dim]No models found — using server default.[/dim]")
        return None

    model_ids = [m["model_id"] for m in models]

    if len(model_ids) == 1:
        console.print(f"[dim]Only one model available:[/dim] [bright_yellow]{model_ids[0]}[/bright_yellow]")
        return model_ids[0]

    table = Table.grid(padding=(0, 2))
    table.add_column(width=3, style="bold cyan")
    table.add_column()
    for i, mid in enumerate(model_ids, 1):
        mtype = models[i - 1].get("model_type", "?")
        table.add_row(str(i), f"[bright_yellow]{mid}[/bright_yellow] [dim]({mtype})[/dim]")
    table.add_row("0", "[dim]Server default[/dim]")

    console.print(Panel(
        table,
        title="[bold bright_cyan]Available Models[/bold bright_cyan]",
        border_style="bright_cyan",
        padding=(1, 2),
    ))

    valid = [str(i) for i in range(len(model_ids) + 1)]
    choice = Prompt.ask(
        "[bold yellow]Select model[/bold yellow]",
        choices=valid,
        default="0",
    )

    idx = int(choice)
    if idx == 0:
        return None
    return model_ids[idx - 1]


# ──────────────────────────────────────────────
# Credential management
# ──────────────────────────────────────────────

def _setup_credentials() -> Optional[Tuple[str, str]]:
    """Interactively prompt for URL and API key. Returns (url, key) or None."""
    console.print(Panel(
        "[bold]Enter your API connection details below.[/bold]\n"
        "[dim]These can be saved locally so you don't have to re-enter them.[/dim]",
        title="[bold bright_cyan]Credentials Setup[/bold bright_cyan]",
        border_style="bright_cyan",
        padding=(1, 2),
    ))

    url = Prompt.ask("[bold yellow]API URL[/bold yellow]").strip()
    if not url:
        _warn("URL cannot be empty.")
        return None

    key = Prompt.ask("[bold yellow]API Key[/bold yellow]").strip()
    if not key:
        _warn("API key cannot be empty.")
        return None

    if Confirm.ask("[bold]Save credentials for future sessions?[/bold]", default=True):
        path = credentials.save(url, key)
        console.print(f"[bright_green]Saved to {path}[/bright_green]")
    else:
        console.print("[dim]Credentials will only be used for this session.[/dim]")

    return url, key


def _ensure_credentials() -> Optional[Tuple[str, str]]:
    """Load saved credentials or prompt interactively. Returns (url, key) or None."""
    saved_url, saved_key = credentials.load()
    if saved_url and saved_key:
        info = Table.grid(padding=(0, 2))
        info.add_column(style="bold")
        info.add_column()
        info.add_row("URL:", f"[bright_cyan]{saved_url}[/bright_cyan]")
        info.add_row("Key:", f"[dim]{saved_key[:12]}...[/dim]")
        console.print(Panel(
            info,
            title="[bold bright_green]Saved Credentials[/bold bright_green]",
            border_style="bright_green",
            padding=(1, 2),
        ))
        return saved_url, saved_key

    console.print("[bold yellow]No saved credentials found.[/bold yellow]")
    return _setup_credentials()


def _credentials_menu(current_url: Optional[str], current_key: Optional[str]) -> Optional[Tuple[str, str]]:
    """Credential management sub-menu. Returns new (url, key) or None to keep current."""
    menu = Table.grid(padding=(0, 2))
    menu.add_column(width=3, style="bold cyan")
    menu.add_column()
    menu.add_row("1", "View current credentials")
    menu.add_row("2", "Set new credentials")
    menu.add_row("3", "Remove saved credentials")
    menu.add_row("0", "Back to main menu")

    console.print(Panel(
        menu,
        title="[bold bright_cyan]Credentials[/bold bright_cyan]",
        border_style="bright_cyan",
        padding=(1, 2),
    ))

    choice = Prompt.ask(
        "[bold yellow]Select option[/bold yellow]",
        choices=["0", "1", "2", "3"],
        default="0",
    )

    if choice == "1":
        saved_url, saved_key = credentials.load()
        info = Table.grid(padding=(0, 2))
        info.add_column(style="bold")
        info.add_column()
        if saved_url and saved_key:
            info.add_row("URL:", f"[bright_cyan]{saved_url}[/bright_cyan]")
            info.add_row("Key:", f"[dim]{saved_key[:12]}...[/dim]")
            info.add_row("File:", f"[dim]{credentials.path()}[/dim]")
            console.print(Panel(info, title="[bold bright_green]Saved Credentials[/bold bright_green]",
                                border_style="bright_green", padding=(1, 2)))
        else:
            console.print("[dim]No credentials saved.[/dim]")
        if current_url and current_key:
            info2 = Table.grid(padding=(0, 2))
            info2.add_column(style="bold")
            info2.add_column()
            info2.add_row("URL:", f"[bright_cyan]{current_url}[/bright_cyan]")
            info2.add_row("Key:", f"[dim]{current_key[:12]}...[/dim]")
            console.print(Panel(info2, title="[bold bright_yellow]Current Session[/bold bright_yellow]",
                                border_style="bright_yellow", padding=(1, 2)))
        return None

    if choice == "2":
        result = _setup_credentials()
        return result

    if choice == "3":
        if credentials.clear():
            console.print("[bright_green]Saved credentials removed.[/bright_green]")
        else:
            console.print("[dim]No saved credentials to remove.[/dim]")
        return None

    return None


# ──────────────────────────────────────────────
# Interactive flows
# ──────────────────────────────────────────────

def _flow_health(client: InferClient) -> None:
    """Health check flow."""
    with console.status("[bold cyan]Connecting to API...[/bold cyan]", spinner="dots"):
        data = client.health()

    info = Table.grid(padding=(0, 2))
    info.add_column(style="bold")
    info.add_column()
    info.add_row("Status:", f"[bold green]{data['status'].upper()}[/bold green]")
    info.add_row("Version:", f"[bright_cyan]{data['version']}[/bright_cyan]")
    info.add_row("Models loaded:", f"[bright_yellow]{data['models_count']}[/bright_yellow]")
    info.add_row("Default model:", f"[bright_magenta]{data['default_model']}[/bright_magenta]")

    console.print(Panel(
        info,
        title="[bold bright_cyan]API Health[/bold bright_cyan]",
        border_style="bright_cyan",
        padding=(1, 2),
    ))

    if data.get("models"):
        table = Table(
            title="Loaded Models",
            box=box.ROUNDED,
            border_style="dim",
            header_style="bold bright_cyan",
        )
        table.add_column("Model", style="bright_yellow")
        table.add_column("Type", style="bright_green")
        table.add_column("Labels", style="dim")
        table.add_column("Ready", justify="center")

        for m in data["models"]:
            ready = "[bold green]\u25cf[/bold green]" if m.get("ready") else "[bold red]\u25cf[/bold red]"
            labels = str(m.get("num_labels", "?"))
            table.add_row(m["model_id"], m.get("model_type", "?"), labels, ready)

        console.print(table)


def _format_gb(value: float) -> str:
    """Format GB values with appropriate precision."""
    if value >= 100:
        return f"{value:.0f} GB"
    elif value >= 10:
        return f"{value:.1f} GB"
    else:
        return f"{value:.2f} GB"


def _create_progress_bar(percent: float, width: int = 20) -> str:
    """Create a text-based progress bar."""
    filled = int((percent / 100) * width)
    empty = width - filled

    if percent >= 90:
        color = "red"
    elif percent >= 70:
        color = "yellow"
    else:
        color = "green"

    bar = "█" * filled + "░" * empty
    return f"[{color}]{bar}[/{color}]"


def _flow_resources(client: InferClient) -> None:
    """Show comprehensive server resources and capacity."""
    try:
        with console.status("[bold cyan]Fetching resources...[/bold cyan]", spinner="dots"):
            data = client.resources()
    except Exception as e:
        _warn(f"Could not fetch resources (API may not support this): {e}")
        return

    gpu = data.get("gpu", {})
    cpu = data.get("cpu", {})
    mem = data.get("memory", {})
    storage = data.get("storage", {})
    system = data.get("system", {})
    capacity = data.get("capacity", {})
    recommendations = data.get("recommendations", {})

    # Create main table
    hw_table = Table(
        show_header=False,
        box=box.SIMPLE_HEAD,
        padding=(0, 1),
        expand=False
    )
    hw_table.add_column("Component", style="bold cyan", width=12)
    hw_table.add_column("Details", style="white", width=50)
    hw_table.add_column("Usage", width=28)

    # GPU Information
    if gpu.get("available"):
        device_type = gpu.get("device_type", "unknown").upper()
        if device_type == "CUDA":
            gpu_icon = "🎮"
        elif device_type == "ROCM":
            gpu_icon = "🔴"
        elif device_type == "MPS":
            gpu_icon = "🍎"
        else:
            gpu_icon = "💻"

        gpu_names = gpu.get("device_names", ["Unknown"])
        gpu_name = gpu_names[0] if gpu_names else "Unknown"

        gpu_details = f"{gpu_icon} [bold bright_green]{gpu_name}[/bold bright_green]"
        if gpu.get("total_memory_gb", 0) > 0:
            gpu_details += f"\n   [cyan]{_format_gb(gpu.get('total_memory_gb', 0))}[/cyan]"

        if gpu.get("total_memory_gb", 0) > 0 and gpu.get("available_memory_gb", 0) > 0:
            mem_percent = ((gpu.get("total_memory_gb", 0) - gpu.get("available_memory_gb", 0)) / gpu.get("total_memory_gb", 1)) * 100
            gpu_usage = f"{_create_progress_bar(mem_percent, 20)}\n[dim green]{_format_gb(gpu.get('available_memory_gb', 0))} free[/dim green]"
        else:
            gpu_usage = "[green]Ready[/green]"

        hw_table.add_row("GPU", gpu_details, gpu_usage)
    else:
        hw_table.add_row("GPU", "💻 [yellow]CPU Only[/yellow]", "—")

    # CPU Information
    proc_name = cpu.get("processor_name", cpu.get("architecture", "Unknown"))
    if len(proc_name) > 40:
        proc_name = proc_name[:37] + "..."

    cpu_details = f"⚡ [bold bright_yellow]{proc_name}[/bold bright_yellow]"
    if cpu.get("physical_cores", 0) > 0:
        cpu_details += f"\n   [cyan]{cpu.get('physical_cores', 0)} cores / {cpu.get('logical_cores', 0)} threads[/cyan]"
        if cpu.get("max_frequency_mhz", 0) > 100:
            cpu_details += f" @ {cpu.get('max_frequency_mhz', 0):.0f} MHz"

    cpu_percent = cpu.get("cpu_percent", 0)
    if cpu_percent > 0:
        cpu_usage = f"{_create_progress_bar(cpu_percent, 20)}\n[dim]{cpu_percent:.1f}% used[/dim]"
    else:
        cpu_usage = "[green]Ready[/green]"

    hw_table.add_row("CPU", cpu_details, cpu_usage)

    # Memory Information
    mem_details = f"🧠 [bold bright_magenta]Memory (RAM)[/bold bright_magenta]"
    mem_details += f"\n   [cyan]{_format_gb(mem.get('total_gb', 0))} total[/cyan]"

    mem_percent = mem.get("percent_used", 0)
    mem_usage = f"{_create_progress_bar(mem_percent, 20)}\n[dim green]{_format_gb(mem.get('available_gb', 0))} free[/dim green]"

    hw_table.add_row("RAM", mem_details, mem_usage)

    # Storage Information
    storage_details = f"💾 [bold bright_blue]Storage (Disk)[/bold bright_blue]"
    storage_details += f"\n   [cyan]{_format_gb(storage.get('total_gb', 0))} total[/cyan]"

    storage_percent = storage.get("percent_used", 0)
    storage_usage = f"{_create_progress_bar(storage_percent, 20)}\n[dim green]{_format_gb(storage.get('available_gb', 0))} free[/dim green]"

    hw_table.add_row("Storage", storage_details, storage_usage)

    # System info
    sys_info = f"{system.get('os_name', '?')} {system.get('os_release', '')}"
    sys_info += f"\n   Python {system.get('python_version', '?')}"
    hw_table.add_row("System", f"🖥️  [dim]{sys_info}[/dim]", "")

    console.print(Panel(
        hw_table,
        title="[bold bright_cyan]⚙️  Server Resources[/bold bright_cyan]",
        border_style="bright_cyan",
        padding=(1, 2),
    ))

    # Capacity and Recommendations side by side
    main_grid = Table.grid(padding=(0, 4))
    main_grid.add_column()
    main_grid.add_column()

    # Capacity
    cap_table = Table(show_header=False, box=None, padding=(0, 1))
    cap_table.add_column("", style="bold", width=18)
    cap_table.add_column("", width=25)

    available = capacity.get("available", False)
    status_color = "bright_green" if available else "bright_red"
    status_text = "✓ Available" if available else "✗ Busy"

    cap_table.add_row("Status:", f"[{status_color}]{status_text}[/{status_color}]")
    cap_table.add_row("Active:", f"[bright_yellow]{capacity.get('active_inferences', 0)}/{capacity.get('max_concurrent', '?')}[/bright_yellow] inferences")

    if not available and capacity.get("reason"):
        cap_table.add_row("Reason:", f"[dim]{capacity.get('reason')}[/dim]")

    # Recommendations
    rec_table = Table(show_header=False, box=None, padding=(0, 1))
    rec_table.add_column("", style="bold green", width=15)
    rec_table.add_column("", width=20)

    device = recommendations.get("device", "cpu").upper()
    if device == "CUDA":
        device_icon = "🎮"
    elif device == "ROCM":
        device_icon = "🔴"
    elif device == "MPS":
        device_icon = "🍎"
    else:
        device_icon = "💻"

    rec_table.add_row("🎯 Device:", f"{device_icon} [bold]{device}[/bold]")
    rec_table.add_row("📦 Batch:", f"[bold cyan]{recommendations.get('batch_size', '?')}[/bold cyan]")
    rec_table.add_row("👷 Workers:", f"[bold yellow]{recommendations.get('num_workers', '?')}[/bold yellow]")
    fp16 = "✓ Yes" if recommendations.get("use_fp16") else "✗ No"
    rec_table.add_row("⚡ FP16:", fp16)

    main_grid.add_row(
        Panel(cap_table, title="[bold]Inference Capacity[/bold]", border_style="cyan", padding=(0, 1)),
        Panel(rec_table, title="[bold]💡 Recommendations[/bold]", border_style="green", padding=(0, 1))
    )

    console.print(main_grid)

    # Notes
    notes = recommendations.get("notes", [])
    if notes:
        notes_text = "\n".join(f"• {note}" for note in notes)
        console.print(Panel(
            notes_text,
            title="[bold]📝 Notes[/bold]",
            border_style="dim",
            padding=(1, 2)
        ))


def _flow_capabilities(client: InferClient) -> None:
    """Show full server capabilities."""
    try:
        with console.status("[bold cyan]Fetching capabilities...[/bold cyan]", spinner="dots"):
            data = client.capabilities()
    except Exception as e:
        _warn(f"Could not fetch capabilities (API may not support this): {e}")
        return

    # Version
    console.print(f"[bold]Server version:[/bold] [bright_cyan]{data.get('version', '?')}[/bright_cyan]")
    console.print()

    # Trained models
    trained = data.get("trained_models", {})
    models = trained.get("models", [])

    if models:
        table = Table(
            title=f"Trained Models ({trained.get('count', 0)})",
            box=box.ROUNDED,
            border_style="bright_cyan",
            header_style="bold bright_cyan",
        )
        table.add_column("Model", style="bright_yellow")
        table.add_column("Type", style="bright_green")
        table.add_column("Labels", style="dim")
        table.add_column("Ready", justify="center")

        for m in models:
            ready = "[bold green]●[/bold green]" if m.get("ready") else "[bold red]●[/bold red]"
            default_marker = " [dim](default)[/dim]" if m.get("model_id") == trained.get("default") else ""
            table.add_row(
                f"{m.get('model_id', '?')}{default_marker}",
                m.get("model_type", "?"),
                str(m.get("num_labels", "?")),
                ready,
            )
        console.print(table)
    else:
        console.print("[dim]No trained models available.[/dim]")

    console.print()

    # Ollama
    ollama = data.get("ollama", {})
    ollama_models = ollama.get("models", [])

    if ollama.get("available"):
        table = Table(
            title=f"Ollama Models ({ollama.get('models_count', 0)})",
            box=box.ROUNDED,
            border_style="bright_magenta",
            header_style="bold bright_magenta",
        )
        table.add_column("Model", style="bright_yellow")
        table.add_column("Size", style="dim", justify="right")

        for m in ollama_models:
            table.add_row(m.get("name", "?"), f"{m.get('size_gb', 0):.1f} GB")
        console.print(table)
    elif ollama.get("enabled"):
        console.print("[bright_magenta]Ollama:[/bright_magenta] [dim]Enabled but not reachable[/dim]")
    else:
        console.print("[bright_magenta]Ollama:[/bright_magenta] [dim]Disabled[/dim]")


def _flow_models(client: InferClient) -> None:
    """List models flow."""
    with console.status("[bold cyan]Fetching models...[/bold cyan]", spinner="dots"):
        models = client.models()

    if not models:
        console.print("[dim]No models available.[/dim]")
        return

    table = Table(
        title="Available Models",
        box=box.ROUNDED,
        border_style="bright_cyan",
        header_style="bold bright_cyan",
    )
    table.add_column("#", style="dim", justify="right")
    table.add_column("Model ID", style="bold bright_yellow")
    table.add_column("Type", style="bright_green")
    table.add_column("Labels", justify="center")
    table.add_column("Classes", style="dim")
    table.add_column("Ready", justify="center")

    for i, m in enumerate(models, 1):
        ready = "[bold green]\u25cf[/bold green]" if m.get("ready") else "[bold red]\u25cf[/bold red]"
        labels_list = m.get("labels", [])
        labels_preview = ", ".join(labels_list[:3])
        if len(labels_list) > 3:
            labels_preview += f" (+{len(labels_list) - 3})"
        table.add_row(
            str(i),
            m["model_id"],
            m.get("model_type", "?"),
            str(m.get("num_labels", "?")),
            labels_preview,
            ready,
        )

    console.print(table)


def _flow_model_info(client: InferClient) -> None:
    """Model info flow — interactively pick a model."""
    model_id = _ask_model(client)
    if model_id is None:
        model_id = Prompt.ask("[bold yellow]Model ID[/bold yellow]").strip()
        if not model_id:
            _warn("No model specified.")
            return

    with console.status(f"[bold cyan]Fetching metadata for '{model_id}'...[/bold cyan]", spinner="dots"):
        info = client.model_info(model_id)

    # Pretty-print key fields
    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="bold")
    meta.add_column()
    meta.add_row("Model ID:", f"[bright_yellow]{info.get('model_id', model_id)}[/bright_yellow]")
    if info.get("base_model"):
        meta.add_row("Base model:", f"[bright_cyan]{info['base_model']}[/bright_cyan]")
    if info.get("labels"):
        meta.add_row("Labels:", f"[dim]{', '.join(info['labels'])}[/dim]")
    if info.get("languages"):
        meta.add_row("Languages:", f"[dim]{', '.join(info['languages'])}[/dim]")

    console.print(Panel(
        meta,
        title=f"[bold bright_yellow]{model_id}[/bold bright_yellow]",
        border_style="bright_cyan",
        padding=(1, 2),
    ))

    # Metrics table
    if info.get("metrics"):
        metrics_table = Table(
            title="Overall Metrics",
            box=box.SIMPLE,
            border_style="dim",
            header_style="bold",
        )
        metrics_table.add_column("Metric", style="bright_magenta")
        metrics_table.add_column("Value", justify="right", style="bright_green")
        for k, v in info["metrics"].items():
            val = f"{v:.4f}" if isinstance(v, float) else str(v)
            metrics_table.add_row(k, val)
        console.print(metrics_table)

    # Per-class metrics
    if info.get("per_class_metrics"):
        pc_table = Table(
            title="Per-Class Metrics",
            box=box.SIMPLE,
            border_style="dim",
            header_style="bold",
        )
        pc_table.add_column("Class", style="bright_yellow")
        pc_table.add_column("Precision", justify="right")
        pc_table.add_column("Recall", justify="right")
        pc_table.add_column("F1", justify="right")
        pc_table.add_column("Support", justify="right", style="dim")
        for cls, vals in info["per_class_metrics"].items():
            pc_table.add_row(
                cls,
                f"{vals.get('precision', 0):.4f}",
                f"{vals.get('recall', 0):.4f}",
                f"{vals.get('f1-score', vals.get('f1', 0)):.4f}",
                str(vals.get("support", "")),
            )
        console.print(pc_table)

    # Hyperparameters
    if info.get("hyperparameters"):
        hp_table = Table(
            title="Hyperparameters",
            box=box.SIMPLE,
            border_style="dim",
            header_style="bold",
        )
        hp_table.add_column("Parameter", style="bright_magenta")
        hp_table.add_column("Value", style="dim")
        for k, v in info["hyperparameters"].items():
            hp_table.add_row(k, str(v))
        console.print(hp_table)


def _flow_models_submenu(client: InferClient) -> None:
    """Models submenu: list models or get detailed info."""
    menu = Table.grid(padding=(0, 2))
    menu.add_column(width=3, style="bold yellow")
    menu.add_column()

    options = [
        ("1", "List all models"),
        ("2", "Model details — metadata, metrics, hyperparameters"),
        ("0", "Back"),
    ]

    for num, desc in options:
        menu.add_row(num, desc)

    console.print(Panel(
        menu,
        title="[bold bright_yellow]📊 Models[/bold bright_yellow]",
        border_style="bright_yellow",
        padding=(1, 2),
    ))

    choice = Prompt.ask(
        "[bold yellow]Select option[/bold yellow]",
        choices=["0", "1", "2"],
        default="1",
    )

    if choice == "1":
        _flow_models(client)
    elif choice == "2":
        _flow_model_info(client)


def _flow_server_status(client: InferClient) -> None:
    """Show combined health and capabilities."""
    # Health
    try:
        with console.status("[bold cyan]Fetching server status...[/bold cyan]", spinner="dots"):
            health = client.health()
    except Exception as e:
        _warn(f"Could not fetch health: {e}")
        return

    info = Table.grid(padding=(0, 2))
    info.add_column(style="bold")
    info.add_column()
    info.add_row("Status:", f"[bold green]{health['status'].upper()}[/bold green]")
    info.add_row("Version:", f"[bright_cyan]{health['version']}[/bright_cyan]")
    info.add_row("Models loaded:", f"[bright_yellow]{health['models_count']}[/bright_yellow]")
    info.add_row("Default model:", f"[bright_magenta]{health['default_model']}[/bright_magenta]")

    console.print(Panel(
        info,
        title="[bold bright_cyan]Server Status[/bold bright_cyan]",
        border_style="bright_cyan",
        padding=(1, 2),
    ))

    # Capabilities overview
    try:
        capabilities = client.capabilities()
    except Exception:
        return

    # Ollama status
    ollama = capabilities.get("ollama", {})
    if ollama.get("available"):
        ollama_status = f"[bright_green]Available[/bright_green] — {ollama.get('models_count', 0)} models"
    elif ollama.get("enabled"):
        ollama_status = "[bright_yellow]Enabled but not reachable[/bright_yellow]"
    else:
        ollama_status = "[dim]Disabled[/dim]"

    console.print(f"\n[bold]Ollama:[/bold] {ollama_status}")

    # Show loaded models
    if health.get("models"):
        table = Table(
            title="Loaded Models",
            box=box.ROUNDED,
            border_style="dim",
            header_style="bold bright_cyan",
        )
        table.add_column("Model", style="bright_yellow")
        table.add_column("Type", style="bright_green")
        table.add_column("Labels", style="dim")
        table.add_column("Ready", justify="center")

        for m in health["models"]:
            ready = "[bold green]●[/bold green]" if m.get("ready") else "[bold red]●[/bold red]"
            labels = str(m.get("num_labels", "?"))
            table.add_row(m["model_id"], m.get("model_type", "?"), labels, ready)

        console.print(table)


def _flow_classify_text(client: InferClient) -> None:
    """Interactive text classification flow."""
    console.print(Panel(
        "[bold]Enter one or more texts to classify.[/bold]\n"
        "[dim]You can enter multiple texts, one per prompt. "
        "Press Enter on an empty line to finish.[/dim]",
        title="[bold bright_cyan]Classify Text[/bold bright_cyan]",
        border_style="bright_cyan",
        padding=(1, 2),
    ))

    texts: list[str] = []
    while True:
        label = f"[bold yellow]Text {len(texts) + 1}[/bold yellow]" if not texts else f"[yellow]Text {len(texts) + 1}[/yellow] [dim](empty to finish)[/dim]"
        text = Prompt.ask(label, default="").strip()
        if not text:
            if not texts:
                _warn("Enter at least one text.")
                continue
            break
        texts.append(text)

    model = _ask_model(client)
    model_label = model or "default"

    with console.status(
        f"[bold cyan]Running inference on {len(texts)} text(s) with model '{model_label}'...[/bold cyan]",
        spinner="dots",
    ):
        if len(texts) == 1:
            result = client.infer(text=texts[0], model=model)
        else:
            result = client.infer(texts=texts, model=model)

    # Results table
    table = Table(
        title=f"Inference Results \u2014 model [bright_yellow]{result['model_id']}[/bright_yellow] ({result['model_type']})",
        box=box.ROUNDED,
        border_style="bright_cyan",
        header_style="bold bright_cyan",
    )
    table.add_column("#", style="dim", justify="right")
    table.add_column("Text", style="white", max_width=60, overflow="ellipsis")
    table.add_column("Label", style="bold bright_green")
    table.add_column("Confidence", justify="right", style="bright_yellow")

    for i, r in enumerate(result["results"], 1):
        conf = f"{r['confidence']:.4f}"
        text_preview = r.get("text", texts[i - 1] if i <= len(texts) else "")
        table.add_row(str(i), text_preview, r["label"], conf)

    console.print(table)

    # Probabilities detail for single text
    if len(result["results"]) == 1 and "probabilities" in result["results"][0]:
        probs = result["results"][0]["probabilities"]
        prob_table = Table(
            title="Class Probabilities",
            box=box.SIMPLE,
            border_style="dim",
            header_style="bold",
        )
        prob_table.add_column("Class", style="bright_magenta")
        prob_table.add_column("Probability", justify="right")
        prob_table.add_column("Bar", min_width=30)

        for label, prob in sorted(probs.items(), key=lambda x: -x[1]):
            bar_len = int(prob * 30)
            filled = "\u2588" * bar_len
            empty = "\u2591" * (30 - bar_len)
            bar = f"[bright_green]{filled}[/bright_green][dim]{empty}[/dim]"
            prob_table.add_row(label, f"{prob:.4f}", bar)

        console.print(prob_table)


def _flow_classify_csv(client: InferClient) -> None:
    """Interactive CSV classification flow."""
    try:
        import pandas as pd
    except ImportError:
        _warn("pandas is required for CSV classification.\nInstall with: pip install infer-client[pandas]")
        return

    # Ask for file path
    while True:
        input_path = Prompt.ask("[bold yellow]Path to CSV file[/bold yellow]").strip()
        if not input_path:
            _warn("File path cannot be empty.")
            continue
        path = Path(input_path).expanduser().resolve()
        if not path.exists():
            _warn(f"File not found: {path}")
            continue
        if not path.suffix.lower() == ".csv":
            if not Confirm.ask(f"[yellow]File '{path.name}' is not a .csv — continue anyway?[/yellow]", default=False):
                continue
        break

    input_path = str(path)

    # Read CSV
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        _warn(f"Could not read CSV: {e}")
        return

    total_rows = len(df)
    columns = df.columns.tolist()

    # Show file info
    info = Table.grid(padding=(0, 2))
    info.add_column(style="bold")
    info.add_column()
    info.add_row("File:", f"[bright_cyan]{path.name}[/bright_cyan]")
    info.add_row("Path:", f"[dim]{path}[/dim]")
    info.add_row("Rows:", f"[bright_yellow]{total_rows:,}[/bright_yellow]")
    info.add_row("Columns:", f"[dim]{', '.join(columns)}[/dim]")

    console.print(Panel(
        info,
        title="[bold bright_cyan]CSV File[/bold bright_cyan]",
        border_style="bright_cyan",
        padding=(1, 2),
    ))

    # Data preview
    preview_table = Table(
        title="Data Preview (first 5 rows)",
        box=box.SIMPLE,
        border_style="dim",
        header_style="bold",
    )
    preview_cols = columns[:5]
    for col in preview_cols:
        preview_table.add_column(col, style="dim", max_width=50, overflow="ellipsis")
    if len(columns) > 5:
        preview_table.add_column(f"(+{len(columns) - 5})", style="dim")

    for _, row in df.head(5).iterrows():
        vals = [str(row[c]) for c in preview_cols]
        if len(columns) > 5:
            vals.append("...")
        preview_table.add_row(*vals)

    console.print(preview_table)
    _separator()

    # Ask for text column
    if len(columns) == 1:
        text_col = columns[0]
        console.print(f"[dim]Only one column available:[/dim] [bright_green]{text_col}[/bright_green]")
    else:
        col_table = Table.grid(padding=(0, 2))
        col_table.add_column(width=3, style="bold cyan")
        col_table.add_column()
        for i, col in enumerate(columns, 1):
            sample = str(df[col].iloc[0])[:60] if len(df) > 0 else ""
            col_table.add_row(str(i), f"[bright_green]{col}[/bright_green] [dim]({sample}...)[/dim]")

        console.print(Panel(
            col_table,
            title="[bold bright_cyan]Select Text Column[/bold bright_cyan]",
            border_style="bright_cyan",
            padding=(1, 2),
        ))

        valid_choices = [str(i) for i in range(1, len(columns) + 1)]
        col_choice = Prompt.ask(
            "[bold yellow]Text column number[/bold yellow]",
            choices=valid_choices,
            default="1",
        )
        text_col = columns[int(col_choice) - 1]

    console.print(f"[dim]Text column:[/dim] [bright_green]{text_col}[/bright_green]")

    # Ask for model
    model = _ask_model(client)
    model_label = model or "default"

    # Ask for batch size
    batch_size = IntPrompt.ask(
        "[bold yellow]Batch size[/bold yellow]",
        default=32,
    )

    # Ask for output path
    default_output = str(path.parent / f"{path.stem}_classified.csv")
    output_path = Prompt.ask(
        "[bold yellow]Output file path[/bold yellow]",
        default=default_output,
    ).strip()

    total_batches = (total_rows + batch_size - 1) // batch_size

    # Confirm
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("Input:", f"[bright_cyan]{path.name}[/bright_cyan]")
    summary.add_row("Rows:", f"[bright_yellow]{total_rows:,}[/bright_yellow]")
    summary.add_row("Text column:", f"[bright_green]{text_col}[/bright_green]")
    summary.add_row("Model:", f"[bright_magenta]{model_label}[/bright_magenta]")
    summary.add_row("Batch size:", f"[dim]{batch_size}[/dim]")
    summary.add_row("Batches:", f"[dim]{total_batches}[/dim]")
    summary.add_row("Output:", f"[bright_green]{output_path}[/bright_green]")

    console.print(Panel(
        summary,
        title="[bold bright_cyan]Classification Summary[/bold bright_cyan]",
        border_style="bright_cyan",
        padding=(1, 2),
    ))

    if not Confirm.ask("[bold]Start classification?[/bold]", default=True):
        console.print("[dim]Cancelled.[/dim]")
        return

    # Progress bar
    start_time = time.time()

    progress = Progress(
        SpinnerColumn("dots", style="bright_cyan"),
        TextColumn("[bold bright_cyan]Classifying[/bold bright_cyan]"),
        BarColumn(bar_width=40, style="bright_blue", complete_style="bright_green", finished_style="bold green"),
        TaskProgressColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("[bright_yellow]{task.fields[rows_done]:,}/{task.fields[rows_total]:,} rows[/bright_yellow]"),
        TextColumn("[dim]|[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]<[/dim]"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    with progress:
        task_id = progress.add_task(
            "classify",
            total=total_rows,
            rows_done=0,
            rows_total=total_rows,
        )

        def _on_batch(processed: int, total: int) -> None:
            progress.update(task_id, completed=processed, rows_done=processed)

        result_path = client.classify_csv(
            input_path=input_path,
            text_column=text_col,
            output_path=output_path,
            model=model,
            batch_size=batch_size,
            on_batch_done=_on_batch,
        )

    elapsed = time.time() - start_time
    rows_per_sec = total_rows / elapsed if elapsed > 0 else 0

    # Completion panel
    done = Table.grid(padding=(0, 2))
    done.add_column(style="bold")
    done.add_column()
    done.add_row("Output:", f"[bold bright_green]{result_path}[/bold bright_green]")
    done.add_row("Rows classified:", f"[bright_yellow]{total_rows:,}[/bright_yellow]")
    done.add_row("Time:", f"[bright_cyan]{elapsed:.1f}s[/bright_cyan]")
    done.add_row("Speed:", f"[bright_magenta]{rows_per_sec:.1f} rows/s[/bright_magenta]")

    console.print()
    console.print(Panel(
        done,
        title="[bold bright_green]Classification Complete[/bold bright_green]",
        border_style="bright_green",
        padding=(1, 2),
    ))


# ──────────────────────────────────────────────
# Ollama flows (via server API)
# ──────────────────────────────────────────────


def _ask_server_ollama_model() -> Optional[str]:
    """Interactively ask for an Ollama model from the server."""
    if _client is None:
        _warn("Not connected to API server.")
        return None

    try:
        with console.status("[bold magenta]Fetching Ollama models from server...[/bold magenta]", spinner="dots"):
            models = _client.ollama_models()
    except Exception as e:
        _warn(f"Could not fetch Ollama models from server: {e}")
        return None

    if not models:
        console.print("[dim]No Ollama models available on the server.[/dim]")
        return None

    model_names = [m["name"] for m in models]

    table = Table.grid(padding=(0, 2))
    table.add_column(width=3, style="bold magenta")
    table.add_column()
    for i, name in enumerate(model_names, 1):
        model_info = models[i - 1]
        size_gb = model_info.get("size", 0) / (1024 ** 3)
        table.add_row(str(i), f"[bright_yellow]{name}[/bright_yellow] [dim]({size_gb:.1f} GB)[/dim]")

    console.print(Panel(
        table,
        title="[bold bright_magenta]Server Ollama Models[/bold bright_magenta]",
        border_style="bright_magenta",
        padding=(1, 2),
    ))

    valid = [str(i) for i in range(1, len(model_names) + 1)]
    choice = Prompt.ask(
        "[bold yellow]Select model[/bold yellow]",
        choices=valid,
        default="1",
    )

    return model_names[int(choice) - 1]


def _flow_ollama_generate() -> None:
    """Interactive Ollama text generation via server API."""
    if _client is None:
        _warn("Not connected to API server.")
        return

    model = _ask_server_ollama_model()
    if model is None:
        return

    console.print(Panel(
        "[bold]Enter your prompt for the LLM.[/bold]\n"
        "[dim]You can optionally provide a system message to guide the model's behavior.[/dim]",
        title="[bold bright_magenta]Generate Text (Server)[/bold bright_magenta]",
        border_style="bright_magenta",
        padding=(1, 2),
    ))

    system = Prompt.ask(
        "[bold yellow]System message[/bold yellow] [dim](optional)[/dim]",
        default="",
    ).strip() or None

    prompt = Prompt.ask("[bold yellow]Prompt[/bold yellow]").strip()
    if not prompt:
        _warn("Prompt cannot be empty.")
        return

    try:
        with console.status(
            f"[bold magenta]Generating with {model} on server...[/bold magenta]",
            spinner="dots",
        ):
            result = _client.ollama_generate(model, prompt, system=system)
        response = result.get("response", "")
    except Exception as e:
        _warn(f"Generation failed: {e}")
        return

    console.print(Panel(
        response,
        title=f"[bold bright_magenta]{model}[/bold bright_magenta]",
        border_style="bright_magenta",
        padding=(1, 2),
    ))


def _flow_ollama_chat() -> None:
    """Interactive Ollama chat via server API."""
    if _client is None:
        _warn("Not connected to API server.")
        return

    model = _ask_server_ollama_model()
    if model is None:
        return

    console.print(Panel(
        "[bold]Start a conversation with the LLM on the server.[/bold]\n"
        "[dim]Type 'exit' or 'quit' to end the conversation.[/dim]",
        title="[bold bright_magenta]Chat Mode (Server)[/bold bright_magenta]",
        border_style="bright_magenta",
        padding=(1, 2),
    ))

    system = Prompt.ask(
        "[bold yellow]System message[/bold yellow] [dim](optional)[/dim]",
        default="",
    ).strip()

    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                console.print("[dim]Ending conversation.[/dim]")
                break

            messages.append({"role": "user", "content": user_input})

            with console.status("[bold magenta]Thinking on server...[/bold magenta]", spinner="dots"):
                result = _client.ollama_chat(model, messages)

            # Update messages with assistant response
            assistant_msg = result.get("response", "")
            messages.append({"role": "assistant", "content": assistant_msg})
            console.print(f"[bold bright_magenta]{model}:[/bold bright_magenta] {assistant_msg}")
            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]Ending conversation.[/dim]")
            break
        except Exception as e:
            _warn(f"Chat error: {e}")
            break


# ──────────────────────────────────────────────
# Welcome Dashboard
# ──────────────────────────────────────────────

def _show_welcome_dashboard(client: InferClient) -> None:
    """Show comprehensive welcome dashboard after connection."""
    console.print()

    # Fetch all data
    health_data = None
    resources_data = None
    models_data = None
    ollama_data = None

    try:
        with console.status("[bold cyan]Loading dashboard...[/bold cyan]", spinner="dots"):
            try:
                health_data = client.health()
            except Exception:
                pass
            try:
                resources_data = client.resources()
            except Exception:
                pass
            try:
                models_data = client.models()
            except Exception:
                pass
            try:
                ollama_data = client.ollama_status()
            except Exception:
                pass
    except Exception:
        pass

    # ─── Server Info Banner ───
    if health_data:
        version = health_data.get("version", "?")
        models_count = health_data.get("models_count", 0)
    else:
        version = "?"
        models_count = 0

    # System info
    if resources_data:
        sys_info = resources_data.get("system", {})
        gpu_info = resources_data.get("gpu", {})
        mem_info = resources_data.get("memory", {})
        cpu_info = resources_data.get("cpu", {})

        os_name = sys_info.get("os_name", "Unknown")

        # Determine device type
        if gpu_info.get("available"):
            device_type = gpu_info.get("device_type", "").upper()
            if device_type == "MPS":
                machine_desc = "🍎 Apple Silicon"
                gpu_mem = gpu_info.get("total_memory_gb", 0)
                if gpu_mem > 0:
                    machine_desc += f" ({gpu_mem:.0f} GB)"
            elif device_type == "CUDA":
                gpu_names = gpu_info.get("device_names", [])
                gpu_name = gpu_names[0] if gpu_names else "NVIDIA GPU"
                machine_desc = f"🎮 {gpu_name}"
            elif device_type == "ROCM":
                machine_desc = "🔴 AMD GPU (ROCm)"
            else:
                machine_desc = "💻 CPU"
        else:
            machine_desc = "💻 CPU Only"

        ram_gb = mem_info.get("total_gb", 0)
        cores = cpu_info.get("physical_cores", 0)
    else:
        os_name = "Unknown"
        machine_desc = "Unknown"
        ram_gb = 0
        cores = 0

    # Welcome message
    welcome_grid = Table.grid(padding=(0, 2))
    welcome_grid.add_column(style="bold")
    welcome_grid.add_column()
    welcome_grid.add_row("Server:", f"[bright_cyan]v{version}[/bright_cyan]")
    welcome_grid.add_row("Machine:", f"[bright_green]{machine_desc}[/bright_green]")
    if ram_gb > 0:
        welcome_grid.add_row("Memory:", f"[bright_yellow]{ram_gb:.0f} GB RAM[/bright_yellow] • [dim]{cores} cores[/dim]")
    welcome_grid.add_row("OS:", f"[dim]{os_name}[/dim]")

    console.print(Panel(
        welcome_grid,
        title="[bold bright_cyan]🖥️  Connected to Inference Server[/bold bright_cyan]",
        border_style="bright_cyan",
        padding=(1, 2),
    ))

    # ─── Quick Resources ───
    if resources_data:
        capacity = resources_data.get("capacity", {})
        recommendations = resources_data.get("recommendations", {})

        available = capacity.get("available", False)
        active = capacity.get("active_inferences", 0)
        max_concurrent = capacity.get("max_concurrent", 4)

        status_icon = "🟢" if available else "🔴"
        status_text = "Ready" if available else "Busy"

        # Compact resource line
        resource_parts = []
        resource_parts.append(f"{status_icon} [bold {'green' if available else 'red'}]{status_text}[/bold {'green' if available else 'red'}]")
        resource_parts.append(f"[dim]Slots: {active}/{max_concurrent}[/dim]")

        if recommendations.get("batch_size"):
            resource_parts.append(f"[dim]Batch: {recommendations.get('batch_size')}[/dim]")

        console.print(" • ".join(resource_parts))
        console.print()

    # ─── Models Table ───
    main_grid = Table.grid(padding=(0, 3))
    main_grid.add_column()
    main_grid.add_column()

    # Classification models
    if models_data:
        model_table = Table(
            box=box.ROUNDED,
            border_style="bright_yellow",
            header_style="bold bright_yellow",
            title_style="bold",
            expand=False,
        )
        model_table.add_column("Model", style="bright_yellow")
        model_table.add_column("Type", style="dim")
        model_table.add_column("Labels", justify="center")

        for m in models_data[:5]:  # Limit to 5 models
            model_table.add_row(
                m.get("model_id", "?"),
                m.get("model_type", "?")[:12],
                str(m.get("num_labels", "?"))
            )

        if len(models_data) > 5:
            model_table.add_row(f"[dim]+{len(models_data) - 5} more...[/dim]", "", "")
    else:
        model_table = Table(box=box.ROUNDED, border_style="dim")
        model_table.add_column("Models")
        model_table.add_row("[dim]No models available[/dim]")

    # Ollama models
    if ollama_data and ollama_data.get("available"):
        ollama_models = ollama_data.get("models", [])
        ollama_table = Table(
            box=box.ROUNDED,
            border_style="bright_magenta",
            header_style="bold bright_magenta",
            expand=False,
        )
        ollama_table.add_column("Ollama Model", style="bright_magenta")
        ollama_table.add_column("Size", style="dim", justify="right")

        if ollama_models:
            for m in ollama_models[:5]:
                size_gb = m.get("size", 0) / (1024 ** 3)
                ollama_table.add_row(m.get("name", "?"), f"{size_gb:.1f} GB")
            if len(ollama_models) > 5:
                ollama_table.add_row(f"[dim]+{len(ollama_models) - 5} more...[/dim]", "")
        else:
            ollama_table.add_row("[dim]No models[/dim]", "")
    else:
        ollama_table = Table(box=box.ROUNDED, border_style="dim")
        ollama_table.add_column("Ollama")
        ollama_table.add_row("[dim]Not available[/dim]")

    main_grid.add_row(
        Panel(model_table, title="[bold]📊 Classification[/bold]", border_style="yellow", padding=(0, 1)),
        Panel(ollama_table, title="[bold]🤖 Generative[/bold]", border_style="magenta", padding=(0, 1))
    )

    console.print(main_grid)
    console.print()


# ──────────────────────────────────────────────
# Main menu
# ──────────────────────────────────────────────

def _main_menu() -> str:
    """Display main menu with two sections: Inference and Settings."""
    # Inference section
    inference_menu = Table.grid(padding=(0, 2))
    inference_menu.add_column(width=3, style="bold bright_green")
    inference_menu.add_column()

    inference_options = [
        ("1", "[bright_green]Classify Text[/bright_green] — Run inference on text(s)"),
        ("2", "[bright_green]Classify CSV[/bright_green] — Batch-classify a dataset"),
        ("3", "[bright_magenta]Ollama Generate[/bright_magenta] — Single prompt generation"),
        ("4", "[bright_magenta]Ollama Chat[/bright_magenta] — Multi-turn conversation"),
        ("5", "[bright_yellow]Models[/bright_yellow] — List & inspect available models"),
    ]

    for num, desc in inference_options:
        inference_menu.add_row(num, desc)

    # Settings section
    settings_menu = Table.grid(padding=(0, 2))
    settings_menu.add_column(width=3, style="bold bright_cyan")
    settings_menu.add_column()

    settings_options = [
        ("6", "Resources — Server hardware & capacity"),
        ("7", "Server Status — Health & capabilities"),
        ("8", "Credentials — Manage API credentials"),
        ("0", "[dim]Exit[/dim]"),
    ]

    for num, desc in settings_options:
        settings_menu.add_row(num, desc)

    # Combined layout
    menu_grid = Table.grid(padding=(1, 4))
    menu_grid.add_column()
    menu_grid.add_column()

    menu_grid.add_row(
        Panel(
            inference_menu,
            title="[bold bright_green]🚀 Inference[/bold bright_green]",
            border_style="bright_green",
            padding=(1, 2),
        ),
        Panel(
            settings_menu,
            title="[bold bright_cyan]⚙️  Settings[/bold bright_cyan]",
            border_style="bright_cyan",
            padding=(1, 2),
        )
    )

    console.print(menu_grid)

    choice = Prompt.ask(
        "[bold yellow]Select option[/bold yellow]",
        choices=["0", "1", "2", "3", "4", "5", "6", "7", "8"],
        default="1",
    )

    return choice


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────


def main() -> None:
    global _client

    _print_banner()

    # Credential setup
    cred = _ensure_credentials()
    if cred is None:
        console.print("[bold red]Cannot proceed without credentials.[/bold red]")
        sys.exit(1)

    url, key = cred
    client = InferClient(base_url=url, api_key=key)
    _client = client  # Set global for Ollama functions

    # Show welcome dashboard instead of simple health check
    _show_welcome_dashboard(client)

    # Main loop
    while True:
        try:
            choice = _main_menu()

            if choice == "0":
                console.print("[dim]Goodbye.[/dim]")
                break

            # Inference options (1-5)
            if choice == "1":
                _flow_classify_text(client)
            elif choice == "2":
                _flow_classify_csv(client)
            elif choice == "3":
                _flow_ollama_generate()
            elif choice == "4":
                _flow_ollama_chat()
            elif choice == "5":
                _flow_models_submenu(client)

            # Settings options (6-8)
            elif choice == "6":
                _flow_resources(client)
            elif choice == "7":
                _flow_server_status(client)
            elif choice == "8":
                result = _credentials_menu(url, key)
                if result is not None:
                    url, key = result
                    client = InferClient(base_url=url, api_key=key)
                    _client = client
                    console.print("[bright_green]Credentials updated for this session.[/bright_green]")
                    _show_welcome_dashboard(client)

            _press_enter()

        except KeyboardInterrupt:
            console.print()
            if Confirm.ask("[bold yellow]Exit Infer Client?[/bold yellow]", default=False):
                console.print("[dim]Goodbye.[/dim]")
                break
            continue
        except Exception as e:
            _warn(str(e))
            _press_enter()


if __name__ == "__main__":
    main()
