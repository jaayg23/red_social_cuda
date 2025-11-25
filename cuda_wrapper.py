"""
CUDA Social Network Wrapper
Compila y ejecuta el código CUDA y parsea los resultados
"""

import subprocess
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class CUDASocialNetwork:
    def __init__(self, cuda_file="social_network.cu", executable="social_network.exe"):
        self.cuda_file = cuda_file
        self.executable = executable
        self.compiled = False
        self.output_cache = None

    def compile(self) -> Tuple[bool, str]:
        """
        Compila el código CUDA usando nvcc
        Returns: (success, message)
        """
        try:
            # Verificar que existe el archivo CUDA
            if not os.path.exists(self.cuda_file):
                return False, f"Archivo {self.cuda_file} no encontrado"

            # Compilar con nvcc
            cmd = f'nvcc -o {self.executable} {self.cuda_file} -std=c++11'
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                self.compiled = True
                return True, "Compilación exitosa"
            else:
                return False, f"Error de compilación:\n{result.stderr}"

        except subprocess.TimeoutExpired:
            return False, "Timeout durante la compilación"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def execute(self) -> Tuple[bool, str]:
        """
        Ejecuta el programa CUDA compilado
        Returns: (success, output)
        """
        try:
            if not self.compiled:
                success, msg = self.compile()
                if not success:
                    return False, msg

            # Ejecutar el programa
            result = subprocess.run(
                f'./{self.executable}' if os.name != 'nt' else self.executable,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                self.output_cache = result.stdout
                return True, result.stdout
            else:
                return False, f"Error de ejecución:\n{result.stderr}"

        except subprocess.TimeoutExpired:
            return False, "Timeout durante la ejecución"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def parse_followers(self, output: str) -> Dict[str, List[Dict]]:
        """Parsea la sección de seguidores"""
        followers_data = {"personas": [], "empresas": []}

        # Buscar sección de personas
        person_match = re.search(r'--- Personas ---\n(.*?)(?=\n---|$)', output, re.DOTALL)
        if person_match:
            for line in person_match.group(1).strip().split('\n'):
                match = re.match(r'(.+?):\s*(\d+)\s*seguidores?', line)
                if match:
                    followers_data["personas"].append({
                        "nombre": match.group(1),
                        "seguidores": int(match.group(2))
                    })

        # Buscar sección de empresas
        company_match = re.search(r'--- Empresas ---\n(.*?)(?=\n=|$)', output, re.DOTALL)
        if company_match:
            for line in company_match.group(1).strip().split('\n'):
                match = re.match(r'(.+?):\s*(\d+)\s*seguidores?', line)
                if match:
                    followers_data["empresas"].append({
                        "nombre": match.group(1),
                        "seguidores": int(match.group(2))
                    })

        return followers_data

    def parse_post_reactions(self, output: str) -> List[Dict]:
        """Parsea las reacciones por publicación"""
        reactions = []

        section = re.search(r'REACCIONES POR PUBLICACION.*?\n(.*?)(?=\n=|$)', output, re.DOTALL)
        if section:
            for line in section.group(1).strip().split('\n'):
                match = re.match(r'Post (\d+).*?:\s*(\d+)\s*likes?,\s*(\d+)\s*dislikes?', line)
                if match:
                    reactions.append({
                        "post_id": int(match.group(1)),
                        "likes": int(match.group(2)),
                        "dislikes": int(match.group(3))
                    })

        return reactions

    def parse_top_posts(self, output: str) -> Dict[str, List[Dict]]:
        """Parsea top publicaciones con más y menos likes"""
        top_data = {"mas_likes": [], "menos_likes": []}

        # Top con MÁS likes
        more_section = re.search(r'--- Top 5 con MAS likes ---\n(.*?)(?=\n---|$)', output, re.DOTALL)
        if more_section:
            for line in more_section.group(1).strip().split('\n'):
                match = re.match(r'\d+\.\s*"(.+?)"\s*-\s*(\d+)\s*likes?', line)
                if match:
                    top_data["mas_likes"].append({
                        "texto": match.group(1),
                        "likes": int(match.group(2))
                    })

        # Top con MENOS likes
        less_section = re.search(r'--- Top 5 con MENOS likes ---\n(.*?)(?=\n=|$)', output, re.DOTALL)
        if less_section:
            for line in less_section.group(1).strip().split('\n'):
                match = re.match(r'\d+\.\s*"(.+?)"\s*-\s*(\d+)\s*likes?', line)
                if match:
                    top_data["menos_likes"].append({
                        "texto": match.group(1),
                        "likes": int(match.group(2))
                    })

        return top_data

    def parse_hashtags(self, output: str) -> Dict:
        """Parsea análisis de hashtags"""
        hashtag_data = {
            "mas_usado": None,
            "conteo": []
        }

        section = re.search(r'ANALISIS DE HASHTAGS.*?\n(.*?)(?=\n=|$)', output, re.DOTALL)
        if section:
            # Hashtag más usado
            most_used = re.search(r'Hashtag mas usado:\s*(#\w+)\s*\((\d+)\s*veces?\)', section.group(1))
            if most_used:
                hashtag_data["mas_usado"] = {
                    "hashtag": most_used.group(1),
                    "cantidad": int(most_used.group(2))
                }

            # Conteo de todos
            for line in section.group(1).split('\n'):
                match = re.match(r'\s*(#\w+):\s*(\d+)\s*publicaciones?', line)
                if match:
                    hashtag_data["conteo"].append({
                        "hashtag": match.group(1),
                        "cantidad": int(match.group(2))
                    })

        return hashtag_data

    def parse_blocked_followers(self, output: str) -> List[Dict]:
        """Parsea seguidores bloqueados"""
        blocked = []

        section = re.search(r'SEGUIDORES BLOQUEADOS.*?\n(.*?)(?=\n=|$)', output, re.DOTALL)
        if section:
            current_user = None
            for line in section.group(1).strip().split('\n'):
                user_match = re.match(r'(.+?):', line)
                blocked_match = re.match(r'\s*-\s*(.+)', line)

                if user_match:
                    current_user = user_match.group(1)
                elif blocked_match and current_user:
                    blocked.append({
                        "usuario": current_user,
                        "bloqueado": blocked_match.group(1)
                    })

        return blocked

    def parse_company_recommendations(self, output: str) -> List[Dict]:
        """Parsea recomendaciones de empresas"""
        recommendations = []

        section = re.search(r'EMPRESAS QUE RECOMIENDAN.*?\n(.*?)(?=\n=|$)', output, re.DOTALL)
        if section:
            for line in section.group(1).strip().split('\n'):
                match = re.match(r'(.+?)\s*recomienda a:\s*(.+)', line)
                if match:
                    recommendations.append({
                        "recomienda": match.group(1),
                        "recomendada": match.group(2)
                    })

        return recommendations

    def get_parsed_data(self) -> Optional[Dict]:
        """
        Ejecuta el programa y retorna todos los datos parseados
        """
        success, output = self.execute()

        if not success:
            return None

        return {
            "seguidores": self.parse_followers(output),
            "reacciones": self.parse_post_reactions(output),
            "top_posts": self.parse_top_posts(output),
            "hashtags": self.parse_hashtags(output),
            "bloqueados": self.parse_blocked_followers(output),
            "recomendaciones": self.parse_company_recommendations(output),
            "output_raw": output
        }


if __name__ == "__main__":
    # Test del wrapper
    network = CUDASocialNetwork()

    print("Compilando código CUDA...")
    success, msg = network.compile()
    print(msg)

    if success:
        print("\nEjecutando programa...")
        success, output = network.execute()
        if success:
            print("✓ Ejecución exitosa")

            # Parsear datos
            data = network.get_parsed_data()
            if data:
                print("\n=== DATOS PARSEADOS ===")
                print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(f"✗ Error: {output}")
