"""
Database Integration - DevHub Trader
Integração do banco de dados com as rotas do Flask
"""

from typing import Dict, Any, List, Optional
import os

# Importar apenas quando necessário para evitar erros circulares
def get_managers():
    """Retorna managers do banco de dados"""
    try:
        from database import db, user_manager, event_manager, analysis_manager, portfolio_manager, diary_manager
        return db, user_manager, event_manager, analysis_manager, portfolio_manager, diary_manager
    except ImportError as e:
        print(f"⚠️ Aviso: Database managers não disponíveis: {e}")
        return None, None, None, None, None, None

# ============================================
# CONFIGURAÇÃO
# ============================================

USE_DATABASE = os.getenv('USE_DATABASE', 'false').lower() == 'true'

print(f"{'✅' if USE_DATABASE else '⚠️'} Banco de dados: {'ATIVADO' if USE_DATABASE else 'DESATIVADO (usando memória)'}")

# ============================================
# WRAPPERS PARA COMPATIBILIDADE
# ============================================

class UserService:
    """Serviço de usuários - abstrai se usa DB ou memória"""
    
    @staticmethod
    def get_user_plan(user_id: str) -> str:
        """Retorna plano do usuário"""
        if USE_DATABASE:
            _, user_manager, _, _, _, _ = get_managers()
            if user_manager:
                user = user_manager.get_user(user_id=user_id)
                return user.get('plan', 'FREE_FOREVER') if user else 'FREE_FOREVER'
            return 'FREE_FOREVER'
        else:
            from main import _ADMIN_USER_PLANS, _PLAN_MIGRATION
            plan = _ADMIN_USER_PLANS.get(user_id, 'FREE_FOREVER')
            # Migrar plano antigo se necessário
            if plan in _PLAN_MIGRATION:
                plan = _PLAN_MIGRATION[plan]
                _ADMIN_USER_PLANS[user_id] = plan
            return plan
    
    @staticmethod
    def get_user_usage(user_id: str) -> Dict[str, int]:
        """Retorna uso atual do usuário"""
        if USE_DATABASE:
            _, user_manager, _, _, _, _ = get_managers()
            if user_manager:
                user = user_manager.get_user(user_id=user_id)
            else:
                user = None
            if user:
                return {
                    'tokens_used': user.get('tokens_used', 0),
                    'portfolios_created': user.get('portfolios_created', 0),
                    'analyses_run': user.get('analyses_run', 0)
                }
            return {'tokens_used': 0, 'portfolios_created': 0, 'analyses_run': 0}
        else:
            from main import _USER_TOKEN_USAGE
            return _USER_TOKEN_USAGE.get(user_id, {
                'tokens_used': 0,
                'portfolios_created': 0,
                'analyses_run': 0
            })
    
    @staticmethod
    def update_user_plan(user_id: str, new_plan: str) -> bool:
        """Atualiza plano do usuário"""
        if USE_DATABASE:
            _, user_manager, _, _, _, _ = get_managers()
            if user_manager:
                return user_manager.update_user(user_id, plan=new_plan)
            return False
        else:
            from main import _ADMIN_USER_PLANS
            _ADMIN_USER_PLANS[user_id] = new_plan
            return True
    
    @staticmethod
    def increment_usage(user_id: str, resource: str, amount: int = 1) -> bool:
        """Incrementa uso de recurso"""
        if USE_DATABASE:
            _, user_manager, _, _, _, _ = get_managers()
            if user_manager:
                return user_manager.increment_usage(user_id, resource, amount)
            return False
        else:
            from main import _USER_TOKEN_USAGE
            if user_id not in _USER_TOKEN_USAGE:
                _USER_TOKEN_USAGE[user_id] = {
                    'tokens_used': 0,
                    'portfolios_created': 0,
                    'analyses_run': 0
                }
            
            resource_map = {
                'tokens': 'tokens_used',
                'portfolios': 'portfolios_created',
                'analyses': 'analyses_run'
            }
            
            key = resource_map.get(resource)
            if key:
                _USER_TOKEN_USAGE[user_id][key] += amount
                return True
            return False
    
    @staticmethod
    def check_limit(user_id: str, resource: str, amount: int = 1) -> Dict[str, Any]:
        """Verifica se usuário pode consumir recurso"""
        if USE_DATABASE:
            _, user_manager, _, _, _, _ = get_managers()
            if user_manager:
                return user_manager.check_limit(user_id, resource, amount)
            return {"allowed": False, "reason": "Database manager não disponível"}
        else:
            # Usar lógica em memória (já implementada em main.py)
            from main import _ADMIN_PLAN_LIMITS, _PLAN_MIGRATION
            
            plan = UserService.get_user_plan(user_id)
            usage = UserService.get_user_usage(user_id)
            
            plan_limits = _ADMIN_PLAN_LIMITS.get(plan, _ADMIN_PLAN_LIMITS['FREE_FOREVER'])
            limit = plan_limits.get(resource, 0)
            
            if limit == -1:
                return {"allowed": True, "remaining": -1}
            
            resource_map = {
                'tokens': 'tokens_used',
                'portfolios': 'portfolios_created',
                'analyses': 'analyses_run'
            }
            
            current = usage.get(resource_map.get(resource, ''), 0)
            remaining = max(0, limit - current)
            
            if amount > remaining:
                return {
                    "allowed": False,
                    "reason": f"Limite de {resource} excedido",
                    "current": current,
                    "limit": limit,
                    "remaining": remaining,
                    "requested": amount
                }
            
            return {
                "allowed": True,
                "current": current,
                "limit": limit,
                "remaining": remaining - amount
            }

class EventService:
    """Serviço de eventos especiais"""
    
    @staticmethod
    def list_events(user_id: str = None) -> List[Dict]:
        """Lista todos os eventos"""
        if USE_DATABASE:
            _, _, event_manager, _, _, _ = get_managers()
            if event_manager:
                return event_manager.get_events(user_id=user_id)
            return []
        else:
            from main import _ADMIN_EVENTS
            return [{'id': k, **v} for k, v in _ADMIN_EVENTS.items()]
    
    @staticmethod
    def create_event(user_id: str, date: str, name: str, **kwargs) -> str:
        """Cria novo evento"""
        if USE_DATABASE:
            _, _, event_manager, _, _, _ = get_managers()
            if event_manager:
                return event_manager.create_event(user_id, date, name, **kwargs)
            return None
        else:
            from main import _ADMIN_EVENTS
            event_id = str(len(_ADMIN_EVENTS) + 1)
            _ADMIN_EVENTS[event_id] = {
                'date': date,
                'name': name,
                **kwargs
            }
            return event_id
    
    @staticmethod
    def delete_event(event_id: str) -> bool:
        """Deleta evento"""
        if USE_DATABASE:
            _, _, event_manager, _, _, _ = get_managers()
            if event_manager:
                return event_manager.delete_event(event_id)
            return False
        else:
            from main import _ADMIN_EVENTS
            if event_id in _ADMIN_EVENTS:
                del _ADMIN_EVENTS[event_id]
                return True
            return False

class AnalysisService:
    """Serviço de análises salvas"""
    
    @staticmethod
    def save_analysis(user_id: str, title: str, analysis_type: str, data: Dict, **kwargs) -> str:
        """Salva análise"""
        if USE_DATABASE:
            _, _, _, analysis_manager, _, _ = get_managers()
            if analysis_manager:
                return analysis_manager.save_analysis(user_id, title, analysis_type, data, **kwargs)
            return None
        else:
            # Salvar em arquivo JSON (fallback)
            import json
            import uuid
            
            analysis_id = str(uuid.uuid4())
            
            os.makedirs('saved_analyses', exist_ok=True)
            filename = f"saved_analyses/{user_id}_{analysis_id}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'id': analysis_id,
                    'user_id': user_id,
                    'title': title,
                    'analysis_type': analysis_type,
                    'data': data,
                    **kwargs
                }, f, indent=2)
            
            return analysis_id
    
    @staticmethod
    def get_analyses(user_id: str, analysis_type: str = None) -> List[Dict]:
        """Lista análises"""
        if USE_DATABASE:
            _, _, _, analysis_manager, _, _ = get_managers()
            if analysis_manager:
                return analysis_manager.get_analyses(user_id, analysis_type)
            return []
        else:
            # Ler de arquivos
            import glob
            import json
            
            pattern = f"saved_analyses/{user_id}_*.json"
            files = glob.glob(pattern)
            
            analyses = []
            for file in files:
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        analysis = json.load(f)
                        if not analysis_type or analysis.get('analysis_type') == analysis_type:
                            analyses.append(analysis)
                except:
                    pass
            
            return sorted(analyses, key=lambda x: x.get('created_at', ''), reverse=True)
    
    @staticmethod
    def delete_analysis(analysis_id: str, user_id: str) -> bool:
        """Deleta análise"""
        if USE_DATABASE:
            _, _, _, analysis_manager, _, _ = get_managers()
            if analysis_manager:
                return analysis_manager.delete_analysis(analysis_id, user_id)
            return False
        else:
            # Deletar arquivo
            filename = f"saved_analyses/{user_id}_{analysis_id}.json"
            if os.path.exists(filename):
                os.remove(filename)
                return True
            return False

class DiaryService:
    """Serviço do Diário Quântico"""
    
    @staticmethod
    def save_entry(user_id: str, entry_date: str, trades_data: Dict, **kwargs) -> str:
        """Salva entrada do diário"""
        if USE_DATABASE:
            _, _, _, _, _, diary_manager = get_managers()
            if diary_manager:
                return diary_manager.save_entry(user_id, entry_date, trades_data, **kwargs)
            return None
        else:
            # Salvar em arquivo
            import json
            import uuid
            
            os.makedirs('diary_entries', exist_ok=True)
            filename = f"diary_entries/{user_id}_{entry_date}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'user_id': user_id,
                    'entry_date': entry_date,
                    'trades_data': trades_data,
                    **kwargs
                }, f, indent=2)
            
            return f"{user_id}_{entry_date}"
    
    @staticmethod
    def get_entry(user_id: str, entry_date: str) -> Optional[Dict]:
        """Busca entrada específica"""
        if USE_DATABASE:
            _, _, _, _, _, diary_manager = get_managers()
            if diary_manager:
                return diary_manager.get_entry(user_id, entry_date)
            return None
        else:
            filename = f"diary_entries/{user_id}_{entry_date}.json"
            if os.path.exists(filename):
                import json
                with open(filename, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
    
    @staticmethod
    def get_entries(user_id: str, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Lista entradas"""
        if USE_DATABASE:
            _, _, _, _, _, diary_manager = get_managers()
            if diary_manager:
                return diary_manager.get_entries(user_id, start_date, end_date)
            return []
        else:
            import glob
            import json
            
            pattern = f"diary_entries/{user_id}_*.json"
            files = glob.glob(pattern)
            
            entries = []
            for file in files:
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        entry = json.load(f)
                        entry_date = entry.get('entry_date')
                        
                        # Filtrar por datas se fornecidas
                        if start_date and entry_date < start_date:
                            continue
                        if end_date and entry_date > end_date:
                            continue
                        
                        entries.append(entry)
                except:
                    pass
            
            return sorted(entries, key=lambda x: x.get('entry_date', ''), reverse=True)

# ============================================
# EXPORTAR SERVIÇOS
# ============================================

__all__ = [
    'USE_DATABASE',
    'UserService',
    'EventService',
    'AnalysisService',
    'DiaryService',
    'get_managers'
]

