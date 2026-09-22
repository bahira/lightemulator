"""
Player Vidéo Softplus - Version EXE distribuable
Utilise OpenCV + NumPy + kernels SPEAR (Softplus, Tanh, Sigmoid, GELU)
Lecture MP4/MKV + Sous-titres SRT
"""

import cv2
import numpy as np
import sys
import time

# --- SPEAR Kernels (formules fermées découvertes par NSGA-II) ---

# Formule tanh: x*(23.965+x²)/(24.362+8.387*x²)
def spear_tanh(x):
    """Applique la fonction tanh element-wise via formule fermée SPEAR."""
    return x * (23.965 + x**2) / (24.362 + 8.387 * x**2)

# Formule sigmoid: 0.5 + 0.530 * tanh_core(0.442*x)
def spear_sigmoid(x):
    """Applique la fonction sigmoid element-wise via formule fermée SPEAR."""
    ax = np.abs(x)
    core = x * (23.965 + x**2) / (24.362 + 8.387 * x**2)
    return 0.5 + 0.530 * core * (1.0 / (1.0 + np.exp(-0.442 * ax)))

# Formule softplus: relu(x) + (0.91586 - 0.19255*|x|) / (1.32910 + 0.58157*|x| + 0.41302*x²)
def spear_softplus(x):
    """Applique la fonction Softplus element-wise via formule fermée SPEAR v3."""
    ax = np.abs(x)
    # relu(x) = max(0, x)
    relu_x = np.maximum(0, x)
    # Partie rationnelle
    num = 0.91586 - 0.19255 * ax
    den = 1.32910 + 0.58157 * ax + 0.41302 * (x ** 2)
    # Eviter la division par zéro
    den = np.where(den == 0, 1e-8, den)
    return relu_x + num / den

# Formule GELU: 0.5*x*(1+tanh(2/π*(x+0.044715))) approximation rationnelle
def spear_gelu(x):
    """Applique la fonction GELU element-wise via approximation SPEAR."""
    # Approximation rationnelle de GELU
    tanh_input = 1.702 * x
    tanh_val = (1 - np.exp(-2 * tanh_input**2)) / (1 + np.exp(-2 * tanh_input**2))
    return 0.5 * x * (1.0 + tanh_val)

# --- Lecture de sous-titres SRT basique ---
def parse_srt(filepath):
    """Parse un fichier SRT basique et renvoie une liste de (start_ms, end_ms, text)."""
    subs = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return subs
    
    blocks = content.split('\n\n')
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3: continue
        # La ligne de timing contient "-->"
        timing_line = None
        text_lines = []
        for line in lines:
            if '-->' in line:
                timing_line = line
            else:
                text_lines.append(line)
        if timing_line:
            parts = timing_line.split('-->')
            if len(parts) == 2:
                start_str = parts[0].strip()
                end_str = parts[1].strip()
                # Parse HH:MM:SS,mmm
                try:
                    h1, m1, s1, ms1 = map(int, start_str.replace(',', ':').split(':'))
                    h2, m2, s2, ms2 = map(int, end_str.replace(',', ':').split(':'))
                    start_ms = h1*3600000 + m1*60000 + s1*1000 + ms1
                    end_ms = h2*3600000 + m2*60000 + s2*1000 + ms2
                    # Text est la concatation des lignes suivantes (jusqu'au prochain block)
                    # On prend tout ce qui reste après le timing dans ce block
                    text = ' '.join([l for l in text_lines if l.strip()])
                    subs.append({
                        'start': start_ms,
                        'end': end_ms,
                        'text': text
                    })
                except:
                    pass
    return subs

# --- Fonction principale ---
def main(video_path, subtitle_path=None):
    # Ouverture vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur: impossible d'ouvrir {video_path}")
        return
    
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Lecture: {video_path}")
    print(f"Resolution: {width}x{height}  FPS: {fps}  Frame total: {total}")
    print("Appuie sur 'Q' pour quitter, 'S' pour changer de kernel, 'ESPACE' pour capture\n")
    
    # Sous-titres
    subs = parse_srt(subtitle_path) if subtitle_path else []
    sub_idx = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Fenêtre
    cv2.namedWindow('Player Softplus', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Player Softplus', width, height + 150) # +100 pour le texte
    
    kernel_names = ['Tanh', 'Softplus', 'Sigmoid', 'GELU']
    current_kernel = 0  # 0=Tanh par défaut
    
    while True:
        long_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Fin de vidéo ou erreur de lecture");
            break
        
        # --- Application du kernel SPEAR ---
        # On transforme en float [0,1] puis applique le kernel, puis on re-redenormalise en [0,255]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        # On étend un peu la plage pour que le kernel soit visible : [-1, 1] -> on maps [0,1] -> [-0.5, 0.5]
        # Simplification : on utilise la gray telle quelle dans [-0.5, 0.5] en soustrayant 0.5
        gray_adj = gray - 0.5  # now in [-0.5, 0.5]
        
        # Application du kernel actuel (selon current_kernel)
        kernel_idx = current_kernel % 4
        if kernel_idx == 0:
            processed = spear_tanh(gray_adj)
        elif kernel_idx == 1:
            processed = spear_softplus(gray_adj)
        elif kernel_idx == 2:
            processed = spear_sigmoid(gray_adj)
        else:  # kernel_idx == 3
            processed = spear_gelu(gray_adj)
        # Remapping en [0, 255]
        processed = np.clip(processed, -1, 1) # safety
        processed = ((processed + 1) / 2 * 255).astype(np.uint8)
        
        # Conversion de retour en BGR pour affichage OpenCV
        disp = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        # Affichage des sous-titres
        if sub_idx < len(subs):
            # Simple temporisation basée sur le nombre de frames
            frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            # Intervalle approximatif (ici on suppose 30 fps pour le calcul)
            fps_eff = 30 
            if frame_count > 0 and (frame_count % (int(fps_eff * 0.5)) == 0):  # tous les 0.5s environ
                # Vérification temps réel plus simple : on simule
                pass
            # On affiche simplement le prochain subtitle disponible si on dépasse son start
            # (implémentation simplifiée : on montre tous les 25 frames)
            if frame_count % 25 == 0 and sub_idx < len(subs):
                # Temps approximatif : frame / fps
                t_sec = frame_count / fps_eff
                if subs[sub_idx]['start'] <= t_sec * 1000 <= subs[sub_idx]['end']:
                    cv2.putText(disp, subs[sub_idx]['text'], (20, height + 30), 
                                font, 0.7, (255, 255, 255), 2)
                    # Préparation au next
                    # On pourrait incrémenter sub_idx si on appuie sur espace, ici on le montre
                    cv2.putText(disp, f"Sub: {subs[sub_idx]['text'][:40]}...", (20, height+60), 
                                font, 0.5, (0, 255, 0), 1)
        
        # Info overlay
        info_text = f"Kernel: {kernel_names[current_kernel]}"#  (Softplus L∞≈4e-3)"
        cv2.putText(disp, info_text, (20, height + 90), 
                    font, 0.6, (0, 255, 0), 2)
        
        # Instructions
        cv2.putText(disp, "Q=Quitter  S=Changer kernel  ESPACE=Capture", 
                    (20, height + 110), font, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Player Softplus', disp)
        
        key = cv2.waitKey(30) & 0xFF  # ~30 fps
        if key == ord('q'):
            break
        elif key == ord('s'):
            current_kernel = (current_kernel + 1) % len(kernel_names)
            print(f"Kernel changé vers: {kernel_names[current_kernel]}")
        elif key == ord(' '):  # Espace
            print(f"Capture frame {int(time.time())}.png")
            # cv2.imwrite(f"capture_{int(time.time())}.png", frame)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Usage: python player_softplus.py video.mp4 [sous_titres.srt]
    vid = sys.argv[1] if len(sys.argv) > 1 else "sample.mp4"
    sub = sys.argv[2] if len(sys.argv) > 2 else None
    main(vid, sub)