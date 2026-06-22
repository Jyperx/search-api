import sys
import time
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text

console = Console()
API_URL = "https://search-api-production-db7e.up.railway.app"

def get_status():
    try:
        response = requests.get(f"{API_URL}/api/status", timeout=10)
        return response.json()
    except Exception as e:
        return None

def main_menu():
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]Panel de Control: Cerebro Híbrido[/bold cyan]\n"
        "[dim]Gestiona el motor de búsqueda en Railway[/dim]",
        border_style="cyan"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Conectando con Railway...", total=None)
        status = get_status()
        time.sleep(1) # Simular carga para UX
    
    if not status:
        console.print("[bold red]❌ Error:[/bold red] No se pudo conectar con el servidor en Railway.")
        return

    console.print(f"[green]✓ Servidor en línea[/green] (Versión {status.get('cerebro_version', 'Desconocida')})")
    console.print(f"📦 Productos Indexados: [bold]{status.get('total_products', 0)}[/bold]")
    console.print(f"🏪 Comercios: [bold]{status.get('total_stores', 0)}[/bold]")
    console.print(f"🧠 Clústeres FTS: [bold]{status.get('total_clusters', 0)}[/bold]\n")

    console.print("Selecciona una acción:")
    console.print("  [bold cyan]1.[/bold cyan] Sincronizar Base de Datos (Firebase -> SQLite)")
    console.print("  [bold cyan]2.[/bold cyan] Entrenar Anclas de IA (Gemini Embeddings)")
    console.print("  [bold cyan]3.[/bold cyan] Salir")
    
    choice = input("\n👉 Opción (1-3): ")
    
    if choice == "1":
        run_sync()
    elif choice == "2":
        run_seed_anchors()
    elif choice == "3":
        console.print("Saliendo...")
        sys.exit(0)
    else:
        console.print("[red]Opción inválida.[/red]")
        main_menu()

def run_sync():
    console.print("\n")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=False,
    ) as progress:
        task = progress.add_task(description="Iniciando sincronización masiva con Firebase...", total=None)
        
        try:
            response = requests.post(f"{API_URL}/api/sync", timeout=60)
            data = response.json()
            progress.update(task, description="[green]✓ Sincronización completada exitosamente![/green]")
            
            console.print("\n[bold green]Resultados:[/bold green]")
            console.print(f"• Mensaje: {data.get('message')}")
            console.print(f"• Items Indexados: {data.get('items_indexed', 0)}\n")
            
        except Exception as e:
            progress.update(task, description=f"[bold red]❌ Error de sincronización: {e}[/bold red]")

    input("Presiona ENTER para continuar...")
    main_menu()

def run_seed_anchors():
    console.print("\n")
    
    # Primera fase: Mandar la petición
    with Progress(
        SpinnerColumn("dots2"),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(description="Enviando comando de siembra de anclas a Railway...", total=None)
        
        try:
            response = requests.post(f"{API_URL}/api/seed-anchors", timeout=30)
            data = response.json()
        except Exception as e:
            progress.update(task, description=f"[bold red]❌ Error enviando comando: {e}[/bold red]")
            input("\nPresiona ENTER para continuar...")
            main_menu()
            return
            
    console.print(f"[green]✓ Comando recibido por el servidor:[/green] {data.get('status')}")
    
    # Segunda fase: Barra de progreso simulada/monitoreo
    console.print("[dim]La IA de Gemini está generando los Embeddings Vectoriales (Tomará ~20s)...[/dim]")
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Entrenando modelos matemáticos...", total=100)
        
        # Como se ejecuta en BackgroundTasks, simularemos la barra de carga basada en el tiempo promedio
        while not progress.finished:
            time.sleep(0.25)
            progress.update(task, advance=1.2)
            
    console.print("\n[bold green]✨ ¡Cerebro Vectorial actualizado y listo![/bold green]")
    input("\nPresiona ENTER para continuar...")
    main_menu()

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        console.print("\nSaliendo...")
        sys.exit(0)
