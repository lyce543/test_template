import requests
import json
import sys

BASE_URL = "http://localhost:8000"


def test_evaluation(ub_id: int):
    print(f"\n{'='*70}")
    print(f" Тестування Evaluation для UB ID: {ub_id}")
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
    
    print("📚 Завантаження історії чату...")
    try:
        history_response = requests.get(f"{BASE_URL}/chat/{ub_id}/history")
        if history_response.status_code == 200:
            history_data = history_response.json()
            message_count = history_data.get('count', 0)
            messages = history_data.get('messages', [])
            
            print(f"   Знайдено повідомлень: {message_count}")
            
            if message_count == 0:
                print("\n⚠️  Історія порожня. Спочатку відправте кілька повідомлень.")
                print(f"   Використайте: python test_agents.py {ub_id} 'ваше повідомлення'")
                return False
            
            print("\n📝 Останні повідомлення:")
            for msg in messages[-3:]:
                user_msg = msg.get('user_message', '')
                ai_msg = msg.get('ai_message', '')
                if user_msg:
                    print(f"   👤 Student: {user_msg[:80]}...")
                if ai_msg:
                    print(f"   🤖 AI: {ai_msg[:80]}...")
        else:
            print(f"❌ Помилка завантаження історії: {history_response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Помилка: {e}")
        return False
    
    print(f"\n{'='*70}")
    print("🔬 Запуск оцінювання...")
    print(f"{'='*70}\n")
    
    try:
        import time
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/chat/{ub_id}/evaluate",
            timeout=120
        )
        
        elapsed = time.time() - start_time
        
        print(f"⏱️  Час виконання: {elapsed:.2f}s")
        print(f"📊 Статус відповіді: {response.status_code}\n")
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ Оцінювання успішно завершено!\n")
            print(f"{'='*70}")
            print(" РЕЗУЛЬТАТ ОЦІНЮВАННЯ")
            print(f"{'='*70}\n")
            
            evaluation_text = data.get('evaluation', '')
            timestamp = data.get('timestamp', '')
            conversation_length = data.get('conversation_length', 0)
            criteria_count = data.get('criteria_count', 0)
            
            print(f"🕐 Час: {timestamp}")
            print(f"💬 Повідомлень в розмові: {conversation_length}")
            print(f"📊 Критеріїв оцінювання: {criteria_count}\n")
            print("📋 Оцінка:\n")
            print(evaluation_text)
            print(f"\n{'='*70}\n")
            
            return True
            
        elif response.status_code == 400:
            error_data = response.json()
            print(f"⚠️  {error_data.get('detail', 'Невідома помилка')}")
            print("\n💡 Можливо, для цього блоку не налаштований evaluation.")
            return False
            
        else:
            print(f"❌ Помилка {response.status_code}")
            print(f"Відповідь: {response.text[:500]}")
            return False
            
    except requests.Timeout:
        print("❌ Таймаут (>120s). Оцінювання займає занадто багато часу.")
        return False
    except Exception as e:
        print(f"❌ Помилка: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n🎓 EdTech AI Platform - Тестування Evaluation\n")
    
    if len(sys.argv) < 2:
        print("❌ Використання: python test_evaluation.py <UB_ID>")
        print("\nПриклад:")
        print("  python test_evaluation.py 12610")
        sys.exit(1)
    
    try:
        ub_id = int(sys.argv[1])
    except ValueError:
        print("❌ UB_ID має бути числом")
        sys.exit(1)
    
    success = test_evaluation(ub_id)
    
    if success:
        print("✅ Тест успішно завершено!")
    else:
        print("\n❌ Тест завершився з помилками")
        sys.exit(1)


if __name__ == "__main__":
    main()