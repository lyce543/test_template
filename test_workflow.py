import requests
import json
import sys
import time

BASE_URL = "http://localhost:8000"


def test_workflow_with_specifications(ub_id: int):
    print(f"\n{'='*70}")
    print(f" Тестування Workflow для UB ID: {ub_id}")
    print(f"{'='*70}\n")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code != 200:
            print("❌ Сервер не відповідає. Запустіть main.py спочатку!")
            return False
    except:
        print("❌ Сервер не запущений. Запустіть: python main.py")
        return False
    
    print("✅ Сервер запущений\n")
    
    print("📚 Завантаження інформації про блок та темплейт...")
    try:
        session_response = requests.get(f"{BASE_URL}/chat/{ub_id}/history")
        if session_response.status_code != 200:
            print(f"❌ Помилка завантаження сесії: {session_response.status_code}")
            return False
        
        session_data = session_response.json()
        print(f"   ✅ Сесія знайдена")
        print(f"   📊 Статус: {session_data.get('status')}")
        print(f"   💬 Повідомлень: {session_data.get('count', 0)}")
        
    except Exception as e:
        print(f"❌ Помилка: {e}")
        return False
    
    print(f"\n{'='*70}")
    print("📤 Відправка тестового повідомлення...")
    print(f"{'='*70}\n")
    
    test_message = input("Введіть тестове повідомлення (або Enter для 'Ембедінг'): ").strip()
    if not test_message:
        test_message = "Ембедінг"
    
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/chat/message",
            json={
                "ub_id": ub_id,
                "content": test_message
            },
            timeout=120
        )
        
        elapsed = time.time() - start_time
        
        print(f"⏱️  Час виконання workflow: {elapsed:.2f}s")
        print(f"📊 Статус відповіді: {response.status_code}\n")
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ Workflow успішно виконано!\n")
            print(f"{'='*70}")
            print(" ВІДПОВІДЬ AI АГЕНТА")
            print(f"{'='*70}\n")
            
            print(f"📝 Type: {data.get('type')}")
            print(f"📌 Title: {data.get('title')}")
            print(f"\n💬 Текст:\n{data.get('text')}\n")
            
            if data.get('additional'):
                print(f"📋 Додаткова інформація:")
                print(json.dumps(data.get('additional'), indent=2, ensure_ascii=False))
            
            print(f"\n{'='*70}\n")
            
            print("🔍 Перевірка оновленої історії...")
            history_response = requests.get(f"{BASE_URL}/chat/{ub_id}/history")
            if history_response.status_code == 200:
                history_data = history_response.json()
                last_messages = history_data.get('messages', [])[-3:]
                
                print("\n📝 Останні 3 повідомлення:")
                for i, msg in enumerate(last_messages, 1):
                    user_msg = msg.get('user_message', '')
                    ai_msg = msg.get('ai_message', '')
                    
                    if user_msg:
                        print(f"\n  {i}. 👤 Student:")
                        print(f"     {user_msg[:100]}...")
                    if ai_msg:
                        print(f"     🤖 AI:")
                        print(f"     {ai_msg[:100]}...")
            
            return True
            
        else:
            print(f"❌ Помилка {response.status_code}")
            print(f"Відповідь: {response.text[:500]}")
            return False
            
    except requests.Timeout:
        print("❌ Таймаут (>120s). Workflow займає занадто багато часу.")
        return False
    except Exception as e:
        print(f"❌ Помилка: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_workflow_tracing(ub_id: int):
    print(f"\n{'='*70}")
    print(f" Перевірка Workflow Tracing")
    print(f"{'='*70}\n")
    
    print("ℹ️  Для перегляду trace відкрийте OpenAI Platform:")
    print("   https://platform.openai.com/traces")
    print("\n   Trace metadata буде містити:")
    print("   - __trace_source__: 'edtech-platform'")
    print(f"   - ub_id: {ub_id}")
    print("   - block_id: <ID блоку>")
    print("   - template_id: <ID темплейту>")


def main():
    print("\n🎓 EdTech AI Platform - Тестування Workflows\n")
    
    if len(sys.argv) < 2:
        print("❌ Використання: python test_workflow.py <UB_ID>")
        print("\nПриклад:")
        print("  python test_workflow.py 12518")
        sys.exit(1)
    
    try:
        ub_id = int(sys.argv[1])
    except ValueError:
        print("❌ UB_ID має бути числом")
        sys.exit(1)
    
    success = test_workflow_with_specifications(ub_id)
    
    if success:
        print("\n" + "="*70)
        test_workflow_tracing(ub_id)
        print("="*70)
        print("\n✅ Тест успішно завершено!")
        
        if input("\nЗапустити evaluation? (y/n): ").strip().lower() == 'y':
            print("\n🔬 Запуск evaluation...")
            import subprocess
            subprocess.run(["python", "test_evaluation.py", str(ub_id)])
    else:
        print("\n❌ Тест завершився з помилками")
        sys.exit(1)


if __name__ == "__main__":
    main()