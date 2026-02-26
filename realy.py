import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import os

# --- 1. تحميل وتدريب الموديل ---
@st.cache_resource
def train_model():
    file_name = 'clean.csv'
    if not os.path.exists(file_name):
        st.error(f"❌ ملف {file_name} غير موجود!")
        st.stop()
        
    data = pd.read_csv(file_name)
    data['price_usd'] = data['price_inr'] / 83 
    
    def find_col(keyword):
        for c in data.columns:
            if keyword.lower() in c.lower(): return c
        return None

    brand_col = find_col('smartphon') or find_col('brand')
    cols = {
        'brand': brand_col,
        'ram': find_col('ram_gb'),
        'storage': find_col('storage_gb'),
        'battery': find_col('battery'),
        'camera': find_col('rear_came'),
        '5g': find_col('has_5g')
    }
    
    features = [cols['brand'], cols['ram'], cols['storage'], cols['battery'], cols['camera'], cols['5g']]
    df_train = data[features].copy()
    
    # تحويل الماركات لأرقام وإضافة خيار "Brand X"
    le = LabelEncoder()
    brands = df_train[cols['brand']].astype(str).unique().tolist()
    if "Brand X (Your Brand)" not in brands:
        brands.append("Brand X (Your Brand)")
    le.fit(brands)
    
    df_train[cols['brand']] = le.transform(df_train[cols['brand']].astype(str))
    
    for col in df_train.columns:
        if col != cols['brand']:
            df_train[col] = pd.to_numeric(df_train[col], errors='coerce').fillna(0).astype(int)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df_train, data['price_usd'])
    
    return model, features, le, data['price_usd'].mean()

model, final_features, label_encoder, avg_market_price = train_model()

# --- 2. الواجهة الرسومية ---
st.set_page_config(page_title="Custom Brand Analyzer", layout="wide")
st.title("🚀 محاكي إطلاق البراند الجديد (Brand X)")
st.subheader("صمم مواصفات جهازك وقارنه بأسعار الماركات العالمية")

with st.sidebar:
    st.header("📋 مواصفات جهازك")
    
    # 1. اختيار الماركة (مع دعم البراند الجديد)
    brand_choice = st.selectbox("الماركة المستهدفة للمقارنة", label_encoder.classes_, index=len(label_encoder.classes_)-1)
    
    # 2. الرام (اختيارات)
    ram = st.selectbox("الرام (RAM GB)", [2, 4, 6, 8, 12, 16, 24], index=3)
    
    # 3. التخزين (اختيارات)
    storage = st.selectbox("التخزين (Storage GB)", [32, 64, 128, 256, 512, 1024], index=2)
    
    # 4. البطارية (اختيارات سهلة)
    battery = st.selectbox("البطارية (Battery mAh)", [3000, 4000, 4500, 5000, 6000, 7000, 10000, 21000], index=3)
    
    # 5. الكاميرا (اختيارات سهلة)
    camera = st.selectbox("الكاميرا الأساسية (MP)", [8, 12, 16, 32, 48, 50, 64, 108, 200], index=5)
    
    is_5g = st.checkbox("دعم شبكة 5G", value=True)
    
    st.markdown("---")
    user_price = st.number_input("سعر البيع المقترح لجهازك ($)", value=250)

if st.button("تحليل فرصة النجاح في السوق"):
    brand_encoded = label_encoder.transform([brand_choice])[0]
    input_data = pd.DataFrame([[brand_encoded, ram, storage, battery, camera, int(is_5g)]], columns=final_features)
    
    # السعر المتوقع بناءً على المنافسين
    fair_market_value = model.predict(input_data)[0]
    
    # حساب كلفة التصنيع التقريبية (Hardware Cost)
    base_cost = (ram * 12) + (storage * 0.15) + (battery * 0.008) + (camera * 0.4) + (40 if is_5g else 0) + 80
    
    # منطق البراند الجديد: إذا كان براند غير معروف، يجب أن يكون السعر أقل من المنافسين بـ 15% للنجاح
    target_for_new_brand = fair_market_value * 0.85
    ratio = user_price / target_for_new_brand
    
    # حساب احتمالية النجاح (ثابتة)
    chance = 100 / (1 + np.exp(9 * (ratio - 1.0)))

    # العرض
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("سعر المنافسين (الماركات المعروفة)", f"${fair_market_value:.2f}")
    col2.metric("كلفة التصنيع التقديرية", f"${base_cost:.2f}")
    col3.metric("صافي ربحك للجهاز الواحد", f"${user_price - base_cost:.2f}")

    color = "#00FF00" if chance > 70 else "#FFA500" if chance > 40 else "#FF4B4B"
    st.markdown(f"### احتمالية نجاح البراند الجديد في السوق: <span style='color:{color}'>{chance:.1f}%</span>", unsafe_allow_html=True)
    st.progress(chance / 100)

    if user_price > fair_market_value:
        st.warning(f"⚠️ تنبيه: سعرك أعلى من الماركات المعروفة بـ {user_price - fair_market_value:.1f}$. كبراند جديد، قد يفضل الزبائن شراء ماركة مشهورة بنفس السعر.")
    elif user_price < base_cost:
        st.error("❌ خطأ تجاري: أنت تبيع بسعر أقل من التكلفة!")
    else:
        st.success("✅ استراتيجية جيدة: سعرك منافس جداً كبداية لبراند جديد.")

    # الرسم البياني
    fig = go.Figure()
    fig.add_trace(go.Bar(x=['سعرك المقترح', 'سعر المنافس المشهور', 'كلفة التصنيع'], 
                         y=[user_price, fair_market_value, base_cost],
                         marker_color=['#00CC96', '#636EFA', '#EF553B']))
    st.plotly_chart(fig, use_container_width=True)