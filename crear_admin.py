"""
Crea el primer código de acceso de administrador.

Ejecutar UNA VEZ al configurar el sistema:
    python crear_admin.py

Guarda el código que muestra en un lugar seguro: con él podrás
crear el resto de códigos (médicos, pacientes) desde la API.
"""

from api.auth.gestor_keys import crear_api_key, listar_keys


if __name__ == "__main__":
    print("=" * 60)
    print("CREACIÓN DEL CÓDIGO DE ADMINISTRADOR")
    print("=" * 60)

    # Comprobar si ya existen códigos
    existentes = listar_keys()
    if existentes:
        print(f"\nAviso: ya existen {len(existentes)} código(s) en el sistema.")
        respuesta = input("¿Crear otro admin de todas formas? (s/n): ")
        if respuesta.lower() != "s":
            print("Cancelado.")
            exit()

    nombre = input("\nNombre del administrador (ej. 'Admin Principal'): ").strip()
    if not nombre:
        nombre = "Admin Principal"

    codigo = crear_api_key(nombre, "admin")

    print("\n" + "=" * 60)
    print("CÓDIGO DE ADMINISTRADOR CREADO")
    print("=" * 60)
    print(f"\nNombre: {nombre}")
    print(f"Código: {codigo}")
    print("\n⚠️  GUARDA ESTE CÓDIGO AHORA. No se mostrará de nuevo.")
    print("   Lo necesitarás para acceder como administrador.")
    print("=" * 60)