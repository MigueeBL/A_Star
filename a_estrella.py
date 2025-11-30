import pygame
import heapq
import time
import json
import random
import sys

# --- Configuración inicial ---
WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 680
PANEL_WIDTH = 600
FPS = 60

# Colores
BLANCO = (245, 245, 245)
NEGRO = (20, 20, 20)
GRIS = (200, 200, 200)
AZUL = (47, 132, 194) # botones y open set
VERDE = (80, 180, 80)          # inicio
ROJO = (200, 70, 70)           # meta
AMARILLO = (240, 230, 120)      # closed set
MORADO = (150, 100, 200)      # camino
VERDE_CLARO = (180, 229, 13)  # control slider

# Grid initial params
DEFAULT_N = 13
MIN_N = 6
MAX_N = 60
DEFAULT_DENSITY = 0.2  # 20% obstacles

# Cell types
FREE = 0
OBSTACLE = 1
START = 2
GOAL = 3

pygame.init()
FONT = pygame.font.SysFont("dejavusans", 16)
FONT_L = pygame.font.SysFont("dejavusans", 20, bold=True)
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("A*")

clock = pygame.time.Clock()

# --- Utilidades UI simples: Button / Slider ---
class Button:
    def __init__(self, rect, text, callback, font=FONT):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.font = font
    def draw(self, surf):
        pygame.draw.rect(surf, AZUL, self.rect, border_radius=6)
        lbl = self.font.render(self.text, True, BLANCO)
        surf.blit(lbl, lbl.get_rect(center=self.rect.center))
    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()

class Slider:
    # horizontal slider from x,y with given width. value range 0..1
    def __init__(self, x,y,w, initial=0.5):
        self.rect = pygame.Rect(x,y,w,20)
        self.value = initial
        self.handle_rect = pygame.Rect(0, y-4, 12, 28)
        self._update_handle()
        self.dragging = False
    def _update_handle(self):
        self.handle_rect.centerx = int(self.rect.x + int(self.value * self.rect.w))
    def draw(self, surf):
        pygame.draw.rect(surf, GRIS, self.rect, border_radius=6)
        pygame.draw.rect(surf, AZUL, (self.rect.x, self.rect.y, int(self.value*self.rect.w), self.rect.h), border_radius=6)
        pygame.draw.rect(surf, VERDE_CLARO, self.handle_rect, border_radius=6)
    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.handle_rect.collidepoint(event.pos):
            self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            relx = max(0, min(self.rect.w, event.pos[0] - self.rect.x))
            self.value = relx / self.rect.w
            self._update_handle()

# --- Grid and A* logic ---
class GridWorld:
    def __init__(self, N=DEFAULT_N, density=DEFAULT_DENSITY):
        self.N = N
        self.density = density
        self.grid = [[FREE for _ in range(self.N)] for _ in range(self.N)]
        self.start = None
        self.goal = None
    def resize(self, N):
        self.N = max(MIN_N, min(MAX_N, int(N)))
        self.clear_all()
    def clear_all(self):
        self.grid = [[FREE for _ in range(self.N)] for _ in range(self.N)]
        self.start = None
        self.goal = None
    def clear_obstacles(self):
        for y in range(self.N):
            for x in range(self.N):
                if self.grid[y][x] == OBSTACLE:
                    self.grid[y][x] = FREE
    def randomize_obstacles(self, density=None):
        if density is None:
            density = self.density
        self.clear_obstacles()
        for y in range(self.N):
            for x in range(self.N):
                if random.random() < density:
                    self.grid[y][x] = OBSTACLE
        # ensure start and goal cells are free if set
        if self.start:
            sx, sy = self.start
            self.grid[sy][sx] = FREE
        if self.goal:
            gx, gy = self.goal
            self.grid[gy][gx] = FREE

# neighbor generation respecting corner cutting
def neighbors_of(node, gw: GridWorld):
    # node = (x,y)
    x,y = node
    N = gw.N
    results = []
    # orthogonals
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    for dx,dy in dirs:
        nx, ny = x+dx, y+dy
        if not (0 <= nx < N and 0 <= ny < N):
            continue
        # diagonal?
        if dx != 0 and dy != 0:
            # anti-corner-cutting: both orthogonal neighbors must be free
            # orthogonal neighbors: (x+dx,y) and (x, y+dy)
            # if any is obstacle -> skip diagonal
            # note: check bounds already true
            if gw.grid[y][x+dx] == OBSTACLE or gw.grid[y+dy][x] == OBSTACLE:
                continue
            cost = 14
        else:
            cost = 10
        results.append(((nx,ny), cost))
    return results

def manhattan_cost(a, b):
    (x1,y1),(x2,y2) = a,b
    return 10 * (abs(x1-x2) + abs(y1-y2))  

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def astar_generator(gw: GridWorld, start, goal):
    """
    Generator that yields progress states for visualization.
    Yields tuples: ('progress', current, open_set, closed_set, g_scores)
                  ('done', path, nodes_expanded, g_cost, time_ms)
                  ('no_path', nodes_expanded, time_ms)
    """
    start_time = time.perf_counter()
    open_heap = []
    counter = 0
    g_score = {start: 0}
    h0 = manhattan_cost(start, goal)
    heapq.heappush(open_heap, (h0, counter, start))
    open_set = {start}
    closed_set = set()
    came_from = {}
    nodes_expanded = 0

    while open_heap:
        f, _, current = heapq.heappop(open_heap)
        if current in closed_set:
            continue
        open_set.discard(current)
        # expand current
        if current == goal:
            t_ms = int((time.perf_counter() - start_time) * 1000)
            path = reconstruct_path(came_from, current)
            yield ('done', path, nodes_expanded, g_score.get(goal, 0), t_ms)
            return
        closed_set.add(current)
        nodes_expanded += 1

        for (nbr, cost) in neighbors_of(current, gw):
            nx, ny = nbr
            # if neighbor is obstacle skip
            if gw.grid[ny][nx] == OBSTACLE:
                continue
            tentative_g = g_score[current] + cost
            if tentative_g < g_score.get(nbr, float('inf')):
                came_from[nbr] = current
                g_score[nbr] = tentative_g
                fscore = tentative_g + manhattan_cost(nbr, goal)
                counter += 1
                heapq.heappush(open_heap, (fscore, counter, nbr))
                open_set.add(nbr)
        # yield progress so UI can animate
        yield ('progress', current, set(open_set), set(closed_set), dict(g_score))
    # no path
    t_ms = int((time.perf_counter() - start_time) * 1000)
    yield ('no_path', nodes_expanded, t_ms)

# --- Drawing helpers ---
def draw_grid(surface, gw: GridWorld, grid_rect, cell_px, open_set=set(), closed_set=set(), path=None):
    # grid_rect: pygame.Rect area where grid is drawn
    N = gw.N
    x0, y0 = grid_rect.x, grid_rect.y
    for y in range(N):
        for x in range(N):
            cell_type = gw.grid[y][x]
            cell_rect = pygame.Rect(x0 + x*cell_px, y0 + y*cell_px, cell_px-1, cell_px-1)
            
            # Determinar color base según el tipo de celda
            if cell_type == FREE:
                color = BLANCO
            elif cell_type == OBSTACLE:
                color = NEGRO
            elif cell_type == START:
                color = VERDE
            elif cell_type == GOAL:
                color = ROJO
            
            # Dibujar celda base
            surface.fill(color, cell_rect)
            
            # Overlays: open/closed sets (solo para celdas FREE)
            if cell_type == FREE:
                if (x,y) in closed_set:
                    surface.fill(AMARILLO, cell_rect.inflate(-2,-2))
                if (x,y) in open_set:
                    surface.fill(AZUL, cell_rect.inflate(-2,-2))
            
            # Ruta (solo para celdas FREE que no son inicio ni meta)
            if path and (x,y) in path and cell_type == FREE:
                surface.fill(MORADO, cell_rect.inflate(-4,-4))
    
    # grid lines
    for i in range(N+1):
        pygame.draw.line(surface, GRIS, (x0 + i*cell_px, y0), (x0 + i*cell_px, y0 + N*cell_px))
        pygame.draw.line(surface, GRIS, (x0, y0 + i*cell_px), (x0 + N*cell_px, y0 + i*cell_px))
        
        
# --- Main app state ---
def main():
    gw = GridWorld(N=DEFAULT_N, density=DEFAULT_DENSITY)
    gw.randomize_obstacles()  # initial obstacles

    status_msg = "Listo. Coloca Incio (S) y Meta (G)"
    mode_set_start = False
    mode_set_goal = False
    placing_obstacles = False
    removing_obstacles = False
    dragging = False

    # grid drawing area - A LA IZQUIERDA
    grid_size_px = min(WINDOW_HEIGHT - 40, WINDOW_WIDTH - PANEL_WIDTH - 40)
    grid_rect = pygame.Rect(20, 20, grid_size_px, grid_size_px)
    cell_px = max(4, grid_rect.width // gw.N)

    # control widgets
    buttons = []
    # Los sliders se posicionarán después de definir todos los botones
    step_mode = False
    running_generator = None
    current_open = set()
    current_closed = set()
    current_path = None
    nodes_expanded_last = 0
    gcost_last = 0
    time_ms_last = 0
    path_length_last = 0
    find_in_progress = False

    # functions bound to buttons
    def btn_randomize():
        gw.randomize_obstacles(density=slider_density.value)
        status = "Obstáculos generados aleatoriamente."
        nonlocal status_msg
        status_msg = status

    def btn_clear():
        gw.clear_all()
        nonlocal current_open, current_closed, current_path, status_msg, find_in_progress, running_generator
        current_open, current_closed, current_path = set(), set(), None
        status_msg = "Limpiado."
        find_in_progress = False
        running_generator = None  # Asegurar que el generador se reinicie

    def btn_find():
        nonlocal running_generator, current_open, current_closed, current_path, nodes_expanded_last, gcost_last, time_ms_last, path_length_last, find_in_progress, status_msg
        if not gw.start or not gw.goal:
            status_msg = "Debes colocar Inicio y Meta antes de buscar."
            return
        if gw.start == gw.goal:
            status_msg = "Inicio == Meta. No se ejecuta búsqueda."
            current_path = [gw.start]
            return
        # validate start/goal on free cells
        sx, sy = gw.start
        gx, gy = gw.goal
        if gw.grid[sy][sx] == OBSTACLE or gw.grid[gy][gx] == OBSTACLE:
            status_msg = "Inicio o Meta está en obstáculo. Libera la celda."
            return
        # start generator
        running_generator = astar_generator(gw, gw.start, gw.goal)
        current_open = set()
        current_closed = set()
        current_path = None
        nodes_expanded_last = 0
        gcost_last = 0
        time_ms_last = 0
        path_length_last = 0
        find_in_progress = True
        status_msg = "Búsqueda iniciada..."

    def btn_export():
        nonlocal current_path, status_msg
        if not current_path:
            status_msg = "No hay ruta para exportar."
            return
        # export as json list of (x,y)
        try:
            with open("ruta_exportada.json", "w", encoding="utf-8") as f:
                json.dump(current_path, f, ensure_ascii=False, indent=2)
            status_msg = "Ruta exportada a 'ruta_exportada.json'."
        except Exception as e:
            status_msg = f"Error exportando: {e}"

    def btn_toggle_step():
        nonlocal step_mode, status_msg
        step_mode = not step_mode
        status_msg = "Modo paso a paso ON." if step_mode else "Modo paso a paso OFF."

    def btn_step_once():
        nonlocal running_generator
        if running_generator is not None:  # Verificar que no sea None
            try:
                ev = next(running_generator)
                handle_generator_event(ev)
            except StopIteration:
                running_generator = None

    def btn_increase_N():
        nonlocal gw, cell_px, grid_rect, status_msg, running_generator
        newN = gw.N + 2
        grid_area_px = min(WINDOW_HEIGHT - 40, WINDOW_WIDTH - PANEL_WIDTH - 40)
        if newN > MAX_N:
            status_msg = f"N máximo = {MAX_N}."
            return
        gw.resize(newN)
        # adjust cell_px
        cell_px = max(4, grid_area_px // gw.N)
        status_msg = f"Nuevo N = {gw.N}."
        running_generator = None  # Reiniciar el generador al cambiar tamaño

    def btn_decrease_N():
        nonlocal gw, cell_px, grid_rect, status_msg, running_generator
        newN = gw.N - 2
        if newN < MIN_N:
            status_msg = f"N mínimo = {MIN_N}."
            return
        gw.resize(newN)
        grid_area_px = min(WINDOW_HEIGHT - 40, WINDOW_WIDTH - PANEL_WIDTH - 40)
        cell_px = max(4, grid_area_px // gw.N)
        status_msg = f"Nuevo N = {gw.N}."
        running_generator = None  # Reiniciar el generador al cambiar tamaño

    # add buttons - TODOS A LA DERECHA
    btn_y = 20
    buttons.append(Button((WINDOW_WIDTH - PANEL_WIDTH + 30, btn_y, 200, 36), "Obstáculos aleatorios", btn_randomize))
    btn_y += 46
    buttons.append(Button((WINDOW_WIDTH - PANEL_WIDTH + 30, btn_y, 200, 36), "Limpiar", btn_clear))
    btn_y += 46
    buttons.append(Button((WINDOW_WIDTH - PANEL_WIDTH + 30, btn_y, 200, 36), "Buscar ruta", btn_find))
    btn_y += 46
    buttons.append(Button((WINDOW_WIDTH - PANEL_WIDTH + 30, btn_y, 200, 36), "Exportar ruta", btn_export))
    btn_y += 46
    buttons.append(Button((WINDOW_WIDTH - PANEL_WIDTH + 30, btn_y, 200, 36), "Modo paso a paso", btn_toggle_step))
    btn_y += 46
    buttons.append(Button((WINDOW_WIDTH - PANEL_WIDTH + 30, btn_y, 200, 36), "Paso único (espacio)", btn_step_once))
    btn_y += 46
    buttons.append(Button((WINDOW_WIDTH - PANEL_WIDTH + 30, btn_y, 95, 36), "N +", btn_increase_N))
    buttons.append(Button((WINDOW_WIDTH - PANEL_WIDTH + 135, btn_y, 95, 36), "N -", btn_decrease_N))

    # Ahora definimos los sliders después de todos los botones
    slider_y = btn_y + 80  # Posición después del último botón
    slider_density = Slider(WINDOW_WIDTH - PANEL_WIDTH + 30, slider_y, 200, initial=gw.density)
    slider_y += 70
    slider_speed = Slider(WINDOW_WIDTH - PANEL_WIDTH + 30, slider_y, 200, initial=0.5)

    # helper to process generator events
    def handle_generator_event(ev):
        nonlocal current_open, current_closed, current_path, nodes_expanded_last, gcost_last, time_ms_last, path_length_last, running_generator, find_in_progress, status_msg
        if ev[0] == 'progress':
            _, current, open_s, closed_s, g_scores = ev
            current_open = open_s
            current_closed = closed_s
            # current_path remains None until done
        elif ev[0] == 'done':
            _, path, nodes_expanded, gcost, time_ms = ev
            current_path = path
            nodes_expanded_last = nodes_expanded
            gcost_last = gcost
            time_ms_last = time_ms
            path_length_last = max(0, len(path)-1)
            status_msg = f"Ruta encontrada. Coste={gcost_last} Steps={path_length_last} Nodos={nodes_expanded_last} Tiempo={time_ms_last}ms"
            # stop generator
            running_generator = None
            find_in_progress = False
        elif ev[0] == 'no_path':
            _, nodes_expanded, time_ms = ev
            current_path = None
            current_open = set()
            current_closed = set()
            nodes_expanded_last = nodes_expanded
            time_ms_last = time_ms
            status_msg = f"SIN RUTA POSIBLE. Nodos expandidos={nodes_expanded_last} Tiempo={time_ms_last}ms"
            running_generator = None
            find_in_progress = False

    # main loop
    running = True
    last_mouse_down = None
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            # UI widgets
            if event.type == pygame.QUIT:
                running = False
            # Buttons and sliders
            for b in buttons:
                b.handle(event)
            slider_density.handle(event)
            slider_speed.handle(event)

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx,my = event.pos
                last_mouse_down = event.pos
                # grid interactions
                if grid_rect.collidepoint(event.pos):
                    # convert to grid coords
                    gx = (event.pos[0] - grid_rect.x) // cell_px
                    gy = (event.pos[1] - grid_rect.y) // cell_px
                    if 0 <= gx < gw.N and 0 <= gy < gw.N:
                        if event.button == 1:  # left click: place obstacle or set start/goal depending on mode
                            if mode_set_start:
                                # set start
                                if gw.grid[gy][gx] != OBSTACLE:
                                    if gw.start:
                                        sx,sy = gw.start
                                        if gw.grid[sy][sx] == START:
                                            gw.grid[sy][sx] = FREE
                                    gw.start = (gx,gy)
                                    gw.grid[gy][gx] = START
                                    status_msg = f"Start colocado en {(gx,gy)}"
                                    mode_set_start = False
                                else:
                                    status_msg = "No se puede colocar Start sobre un obstáculo."
                            elif mode_set_goal:
                                if gw.grid[gy][gx] != OBSTACLE:
                                    if gw.goal:
                                        gx0,gy0 = gw.goal
                                        if gw.grid[gy0][gx0] == GOAL:
                                            gw.grid[gy0][gx0] = FREE
                                    gw.goal = (gx,gy)
                                    gw.grid[gy][gx] = GOAL
                                    status_msg = f"Goal colocado en {(gx,gy)}"
                                    mode_set_goal = False
                                else:
                                    status_msg = "No se puede colocar Goal sobre un obstáculo."
                            else:
                                # place obstacle (toggle)
                                if gw.grid[gy][gx] == OBSTACLE:
                                    gw.grid[gy][gx] = FREE
                                elif gw.grid[gy][gx] in (START, GOAL):
                                    # don't overwrite start/goal accidentally
                                    status_msg = "Usa los botones para mover Start/Goal."
                                else:
                                    gw.grid[gy][gx] = OBSTACLE
                                placing_obstacles = True
                        elif event.button == 3:  # right click: remove obstacle
                            if gw.grid[gy][gx] == OBSTACLE:
                                gw.grid[gy][gx] = FREE
                            removing_obstacles = True
                else:
                    # clicks outside grid may toggle set-start/set-goal when clicking dedicated zones
                    pass

            elif event.type == pygame.MOUSEBUTTONUP:
                placing_obstacles = False
                removing_obstacles = False
                last_mouse_down = None

            elif event.type == pygame.MOUSEMOTION:
                if placing_obstacles and grid_rect.collidepoint(event.pos):
                    gx = (event.pos[0] - grid_rect.x) // cell_px
                    gy = (event.pos[1] - grid_rect.y) // cell_px
                    if 0 <= gx < gw.N and 0 <= gy < gw.N and gw.grid[gy][gx] == FREE:
                        gw.grid[gy][gx] = OBSTACLE
                if removing_obstacles and grid_rect.collidepoint(event.pos):
                    gx = (event.pos[0] - grid_rect.x) // cell_px
                    gy = (event.pos[1] - grid_rect.y) // cell_px
                    if 0 <= gx < gw.N and 0 <= gy < gw.N and gw.grid[gy][gx] == OBSTACLE:
                        gw.grid[gy][gx] = FREE

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    mode_set_start = True
                    mode_set_goal = False
                    status_msg = "Click en una celda para colocar START."
                elif event.key == pygame.K_g:
                    mode_set_goal = True
                    mode_set_start = False
                    status_msg = "Click en una celda para colocar GOAL."
                elif event.key == pygame.K_SPACE:
                    # step once
                    btn_step_once()
                elif event.key == pygame.K_ESCAPE:
                    mode_set_start = False
                    mode_set_goal = False
                    status_msg = "Modos Start/Goal cancelados."

        # update derived params
        grid_size_px = min(WINDOW_HEIGHT - 40, WINDOW_WIDTH - PANEL_WIDTH - 40)
        grid_rect.width = grid_rect.height = grid_size_px
        cell_px = max(4, grid_rect.width // gw.N)
        
        gw.density = slider_density.value

        # process generator if running and not in step_mode
        if running_generator is not None and not step_mode:  # Verificar que no sea None
            steps_pf = 1 + int(slider_speed.value * 40)
            for _ in range(steps_pf):
                if running_generator is not None:  # Verificar nuevamente dentro del loop
                    try:
                        ev = next(running_generator)
                        handle_generator_event(ev)
                    except StopIteration:
                        running_generator = None
                        break
                else:
                    break

        # draw background
        screen.fill((230,230,230))
        # draw grid area background - IZQUIERDA
        pygame.draw.rect(screen, NEGRO, grid_rect, 2)
        # draw the grid cells
        draw_grid(screen, gw, grid_rect, cell_px, open_set=current_open, closed_set=current_closed, path=current_path)
        
        # draw right panel
        panel_x = WINDOW_WIDTH - PANEL_WIDTH
        pygame.draw.rect(screen, (240,240,240), (panel_x, 0, PANEL_WIDTH, WINDOW_HEIGHT))
        
        # draw buttons - TODOS A LA DERECHA
        for b in buttons:
            b.draw(screen)
        
        # SLIDERS DEBAJO DE LOS BOTONES
        slider_start_y = btn_y + 60  # Posición después del último botón
        lbl = FONT.render("Densidad (obstáculos):", True, NEGRO)
        screen.blit(lbl, (panel_x + 30, slider_start_y - 10))
        slider_density.draw(screen)
        
        lbl2 = FONT.render("Velocidad (exec):", True, NEGRO)
        screen.blit(lbl2, (panel_x + 30, slider_start_y + 60))
        slider_speed.draw(screen)

        # CONTROLES - AL LADO DERECHO DE LOS BOTONES
        controls_x = panel_x + 250  # Más a la derecha
        controls_y = 20
        hints = [
            "Controles:",
            "- S: seleccionar Inicio (click)",
            "- G: seleccionar Meta (click)",
            "- Click izquierdo: poner obstáculo",
            "- Click derecho: quitar obstáculo",
            "- Espacio: paso único",
            "(cuando el modo paso a paso está ON)",
            "- Modo paso a paso: click en el botón",
        ]
        for h in hints:
            screen.blit(FONT.render(h, True, NEGRO), (controls_x, controls_y))
            controls_y += 24

        # MÉTRICAS - DEBAJO DE LOS SLIDERS
        metrics_x = panel_x + 30
        metrics_y = slider_start_y + 120  # Posición después de los sliders
        screen.blit(FONT_L.render("Métricas:", True, NEGRO), (metrics_x, metrics_y))
        metrics_y += 28
        screen.blit(FONT.render(f"N = {gw.N}", True, NEGRO), (metrics_x, metrics_y)); metrics_y += 20
        screen.blit(FONT.render(f"Densidad ≈ {gw.density:.2f}", True, NEGRO), (metrics_x, metrics_y)); metrics_y += 20
        screen.blit(FONT.render(f"Nodos expandidos: {nodes_expanded_last}", True, NEGRO), (metrics_x, metrics_y)); metrics_y += 20
        screen.blit(FONT.render(f"Coste total (g): {gcost_last}", True, NEGRO), (metrics_x, metrics_y)); metrics_y += 20
        screen.blit(FONT.render(f"Longitud (pasos): {path_length_last}", True, NEGRO), (metrics_x, metrics_y)); metrics_y += 20
        screen.blit(FONT.render(f"Tiempo búsqueda: {time_ms_last} ms", True, NEGRO), (metrics_x, metrics_y)); metrics_y += 20

        # draw info/status
        screen.blit(FONT.render("Estado: " + status_msg, True, NEGRO), (20, WINDOW_HEIGHT - 32))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()