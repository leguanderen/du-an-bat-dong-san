"""Gọi thẳng mọi hàm render với streamlit giả, để bắt lỗi trước khi demo."""
import sys, types, traceback
import pandas as pd, numpy as np

goi = []
class _RERUN(Exception): pass
_n = None  # gán sau khi định nghĩa
class Ctx:
    def __enter__(self): return self
    def __exit__(self,*a): return False
class Col(Ctx):
    def __getattr__(self,n): return _n(n)
st = types.ModuleType("streamlit")
def _n(name):
    def f(*a,**k):
        goi.append(name)
        if name in ("selectbox","radio"):
            o=(a[1] if len(a)>1 else k.get("options")) or []
            o=list(o)
            if k.get("index",0) is None: return None
            return o[k.get("index",0)] if o else None
        if name=="columns":
            n=a[0] if a else 2
            return [Col() for _ in range(n if isinstance(n,int) else len(n))]
        if name in ("toggle","checkbox"): return k.get("value",False)
        if name=="slider": return k.get("value", a[3] if len(a)>3 else 1.0)
        if name=="number_input": return k.get("value")
        if name in ("text_input","text_area"):
            return globals().get("BOM_CAU") or (k.get("value","") or "")
        if name=="multiselect": return k.get("default",[])
        if name in ("container","expander","form","spinner","status","sidebar","empty","chat_message"): return Ctx()
        if name=="chat_input": return globals().get("BOM_CAU") or None
        if name=="rerun": raise _RERUN()
        if name=="tabs": return [Ctx() for _ in (a[0] if a else [1])]
        if name=="button": return False
        if name=="form_submit_button": return bool(globals().get("BOM_GUI"))
        if name=="pydeck_chart": return None
        return None
    return f
for n in ("markdown caption write info warning success error metric dataframe table "
          "selectbox radio columns toggle checkbox slider number_input multiselect "
          "container expander form spinner status empty tabs button form_submit_button "
          "chat_message chat_input rerun pydeck_chart plotly_chart altair_chart image divider subheader header title "
          "text_input text_area date_input color_picker json code progress").split():
    setattr(st, n, _n(n))
st.session_state = {}
st.column_config = types.SimpleNamespace(
    NumberColumn=lambda *a,**k: None, LinkColumn=lambda *a,**k: None,
    TextColumn=lambda *a,**k: None, ProgressColumn=lambda *a,**k: None)
def _cd(*a, **k):
    if a and callable(a[0]): return a[0]
    return lambda f: f
st.cache_data = _cd; st.cache_resource = _cd
st.set_page_config = lambda *a,**k: None
comp = types.ModuleType("streamlit.components")
v1 = types.ModuleType("streamlit.components.v1")
v1.html = lambda *a, **k: None
v1.iframe = lambda *a, **k: None
v1.declare_component = lambda *a, **k: (lambda *aa, **kk: None)
comp.v1 = v1
st.components = comp
sys.modules["streamlit"] = st
sys.modules["streamlit.components"] = comp
sys.modules["streamlit.components.v1"] = v1

from pathlib import Path as _P
_G = _P(__file__).resolve().parent.parent
sys.path[:0] = [str(d) for d in
                (_G / "app_moi", _G, _G / "pipeline") if d.is_dir()]
import app

loi = 0
for ten in ("render_chungcu","render_nhadat","render_ban_do","render_tim_nha"):
    try:
        getattr(app, ten)()
        print(f"  OK   {ten}")
    except Exception as e:
        loi += 1
        print(f"  LỖI  {ten}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=4)
print("\nsố lỗi:", loi)
