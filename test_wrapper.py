"""
Script de prueba para verificar que el wrapper funciona correctamente
"""

from cuda_wrapper import CUDASocialNetwork
import json

def main():
    print("=" * 60)
    print("  TEST - CUDA Social Network Wrapper")
    print("=" * 60)
    print()

    # Crear instancia
    network = CUDASocialNetwork()

    # Probar compilación
    print("1️⃣  Compilando código CUDA...")
    print("-" * 60)
    success, msg = network.compile()

    if success:
        print("✅ " + msg)
        print()
    else:
        print("❌ " + msg)
        print()
        print("NOTA: Si no tienes GPU NVIDIA o CUDA instalado,")
        print("      esto es normal. Puedes usar Google Colab.")
        return

    # Probar ejecución
    print("2️⃣  Ejecutando programa CUDA...")
    print("-" * 60)
    success, output = network.execute()

    if success:
        print("✅ Ejecución exitosa")
        print()
        print("Primeras líneas del output:")
        print("-" * 60)
        lines = output.split('\n')[:20]
        print('\n'.join(lines))
        print("...")
        print()
    else:
        print("❌ " + output)
        return

    # Probar parsing
    print("3️⃣  Parseando datos...")
    print("-" * 60)
    data = network.get_parsed_data()

    if data:
        print("✅ Datos parseados correctamente")
        print()

        # Mostrar resumen
        print("📊 RESUMEN DE DATOS:")
        print("-" * 60)

        # Seguidores
        if data['seguidores']['personas']:
            print(f"👥 Personas: {len(data['seguidores']['personas'])}")
            top_persona = max(data['seguidores']['personas'], key=lambda x: x['seguidores'])
            print(f"   → Top: {top_persona['nombre']} ({top_persona['seguidores']} seguidores)")

        if data['seguidores']['empresas']:
            print(f"💼 Empresas: {len(data['seguidores']['empresas'])}")
            top_empresa = max(data['seguidores']['empresas'], key=lambda x: x['seguidores'])
            print(f"   → Top: {top_empresa['nombre']} ({top_empresa['seguidores']} seguidores)")

        # Publicaciones
        if data['reacciones']:
            print(f"📝 Publicaciones: {len(data['reacciones'])}")
            total_likes = sum(p['likes'] for p in data['reacciones'])
            print(f"   → Total likes: {total_likes}")

        # Hashtags
        if data['hashtags']['conteo']:
            print(f"#️⃣  Hashtags: {len(data['hashtags']['conteo'])}")
            if data['hashtags']['mas_usado']:
                top_tag = data['hashtags']['mas_usado']
                print(f"   → Más usado: {top_tag['hashtag']} ({top_tag['cantidad']} veces)")

        # Bloqueos
        if data['bloqueados']:
            print(f"🚫 Bloqueos: {len(data['bloqueados'])}")

        # Recomendaciones
        if data['recomendaciones']:
            print(f"💡 Recomendaciones empresariales: {len(data['recomendaciones'])}")

        print()
        print("=" * 60)
        print("✅ ¡Todo funciona correctamente!")
        print("=" * 60)
        print()
        print("Ahora puedes ejecutar la interfaz Streamlit:")
        print("  streamlit run app.py")
        print()

    else:
        print("❌ Error al parsear datos")


if __name__ == "__main__":
    main()
