import requests
import json
import sys
import time

BASE_URL = "http://localhost:8000"


def test_multi_step_workflow(ub_id: int):
    print(f"\n{'='*70}")
    print(f" Тестування Multi-Step Workflow для UB ID: {ub_id}")
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
    
    print("📊 Поточний стан workflow:")
    try:
        state_response = requests.get(f"{BASE_URL}/chat/{ub_id}/state")
        if state_response.status_code == 200:
            state = state_response.json()
            print(f"   Питання: {state.get('current_question_index', 0) + 1}")
            print(f"   Follow-ups: {state.get('follow_up_count', 0)}/3")
            print(f"   Статус: {state.get('status', 'unknown')}")
            print(f"   Всього відповідей: {len(state.get('answers', []))}\n")
        else:
            print("   ⚠️  Стан не знайдено (новий чат)\n")
    except Exception as e:
        print(f"   ⚠️  Помилка завантаження стану: {e}\n")
    
    print(f"{'='*70}")
    print("🎯 Сценарій тестування")
    print(f"{'='*70}\n")
    
    test_scenarios = [
        {
            "step": "1. Неправильна відповідь",
            "message": "не знаю",
            "expected": "має бути follow-up або перехід далі"
        },
        {
            "step": "2. Часткова відповідь",
            "message": "кодування",
            "expected": "має просити уточнення"
        },
        {
            "step": "3. Повна правильна відповідь",
            "message": "Позиційне кодування додає інформацію про послідовність слів у реченні",
            "expected": "має перейти до наступного питання"
        }
    ]
    
    print("Оберіть сценарій:")
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"{i}. {scenario['step']}")
        print(f"   Повідомлення: '{scenario['message']}'")
        print(f"   Очікується: {scenario['expected']}\n")
    
    print("4. Власне повідомлення")
    print("5. Запустити всі сценарії послідовно\n")
    
    choice = input("Ваш вибір (1-5): ").strip()
    
    if choice in ["1", "2", "3"]:
        scenario = test_scenarios[int(choice) - 1]
        send_and_analyze(ub_id, scenario["message"], scenario["step"])
    
    elif choice == "4":
        custom_message = input("\nВведіть ваше повідомлення: ").strip()
        if custom_message:
            send_and_analyze(ub_id, custom_message, "Власне повідомлення")
    
    elif choice == "5":
        print("\n🚀 Запуск всіх сценаріїв...\n")
        for scenario in test_scenarios:
            print(f"\n{'─'*70}")
            send_and_analyze(ub_id, scenario["message"], scenario["step"])
            time.sleep(1)
    
    else:
        print("❌ Невірний вибір")
        return False
    
    print(f"\n{'='*70}")
    print("📊 Фінальний стан workflow")
    print(f"{'='*70}\n")
    
    try:
        state_response = requests.get(f"{BASE_URL}/chat/{ub_id}/state")
        if state_response.status_code == 200:
            state = state_response.json()
            print(f"✅ Стан успішно завантажено:\n")
            print(json.dumps(state, indent=2, ensure_ascii=False))
        else:
            print("⚠️  Не вдалося завантажити стан")
    except Exception as e:
        print(f"❌ Помилка: {e}")
    
    return True


def send_and_analyze(ub_id: int, message: str, step_name: str):
    print(f"{'─'*70}")
    print(f"{step_name}")
    print(f"{'─'*70}")
    print(f"📤 Відправка: '{message}'\n")
    
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/chat/message",
            json={"ub_id": ub_id, "content": message},
            timeout=120
        )
        
        elapsed = time.time() - start_time
        
        print(f"⏱️  Час відповіді: {elapsed:.2f}s")
        print(f"📊 Статус: {response.status_code}\n")
        
        if response.status_code == 200:
            data = response.json()
            ai_text = data.get('text', '')
            
            print(f"✅ Відповідь AI:")
            print(f"   {ai_text}\n")
            
            state_response = requests.get(f"{BASE_URL}/chat/{ub_id}/state")
            if state_response.status_code == 200:
                state = state_response.json()
                print(f"📊 Оновлений стан:")
                print(f"   Питання: {state.get('current_question_index', 0) + 1}")
                print(f"   Follow-ups: {state.get('follow_up_count', 0)}/3")
                print(f"   Статус: {state.get('status', 'unknown')}")
                
                if state.get('answers'):
                    last_answer = state['answers'][-1]
                    if 'evaluation' in last_answer:
                        eval_data = last_answer['evaluation']
                        print(f"\n🔬 Оцінка останньої відповіді:")
                        print(f"   Правильна: {eval_data.get('is_correct', False)}")
                        print(f"   Часткова: {eval_data.get('is_partial', False)}")
                        if eval_data.get('missing_concepts'):
                            print(f"   Не вистачає: {eval_data.get('missing_concepts')}")
            
            return True
            
        else:
            print(f"❌ Помилка {response.status_code}")
            print(f"Відповідь: {response.text[:300]}")
            return False
            
    except requests.Timeout:
        print("❌ Таймаут (>120s)")
        return False
    except Exception as e:
        print(f"❌ Помилка: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_workflow_info():
    print("\n" + "="*70)
    print(" ℹ️  Інформація про Multi-Step Workflow")
    print("="*70 + "\n")
    
    print("🔄 Як працює workflow:\n")
    print("1. Система ставить питання зі списку specifications")
    print("2. Evaluator Agent аналізує відповідь студента:")
    print("   • Перевіряє наявність key_concepts")
    print("   • Визначає: правильна / часткова / неправильна")
    print("3. Якщо відповідь часткова → follow-up (до 3 разів)")
    print("4. Якщо правильна або досягнуто 3 follow-ups → наступне питання")
    print("5. Стан зберігається в Xano між кожним запитом\n")
    
    print("📋 Структура стану:")
    print("   • current_question_index - номер поточного питання")
    print("   • follow_up_count - лічильник уточнень (0-3)")
    print("   • answers[] - всі відповіді з оцінками")
    print("   • status - активний / завершений\n")
    
    print("🎯 Трейсинг:")
    print("   Відкрийте: https://platform.openai.com/traces")
    print("   Фільтр: __trace_source__ = 'edtech-platform'\n")


def main():
    print("\n🎓 EdTech AI Platform - Multi-Step Workflow Test\n")
    
    if len(sys.argv) < 2:
        print("❌ Використання: python test_workflow.py <UB_ID>")
        print("\nПриклад:")
        print("  python test_workflow.py 12610")
        print("\nДля інформації:")
        print("  python test_workflow.py info")
        sys.exit(1)
    
    if sys.argv[1] == "info":
        show_workflow_info()
        sys.exit(0)
    
    try:
        ub_id = int(sys.argv[1])
    except ValueError:
        print("❌ UB_ID має бути числом")
        sys.exit(1)
    
    success = test_multi_step_workflow(ub_id)
    
    if success:
        print("\n✅ Тест завершено!")
        print("\n💡 Корисні команди:")
        print(f"  curl http://localhost:8000/chat/{ub_id}/state")
        print(f"  python test_workflow.py {ub_id}")
        print("  python test_workflow.py info")
    else:
        print("\n❌ Тест завершився з помилками")
        sys.exit(1)


if __name__ == "__main__":
    main()