# churn_prediction

## İş Problemi
Şirketi terk edecek müşterileri tahmin 
edebilecek bir makine öğrenmesi modeli 
geliştirilmesi beklenmektedir.

##  Veri Seti Hikayesi
60000 gözlemden ve 66 değişkenden oluşmaktadır.
Bağımsız değişkenler müşterilere ilişkin bilgiler barındırmaktadır.
Bağımlı değişken müşteri terk durumunu ifade etmektedir.

## Değişkenler
- year, month → Verinin ait olduğu yıl ve ay.
- user_account_id → Kullanıcı kimliği (muhtemelen anonimleştirilmiş).
- user_lifetime → Kullanıcının sisteme kayıt olduğu günden itibaren geçen süre.
- user_intake → Kullanıcının başlangıç paketi, kampanya veya promosyon gibi bilgileri içerebilir.
  
- user_no_outgoing_activity_in_days → Kullanıcının kaç gündür hiçbir giden aktivitesinin (çağrı, SMS vb.) olmadığı.
- user_account_balance_last → Kullanıcının en son hesap bakiyesi.
- user_spendings → Toplam harcamalar.

- user_has_outgoing_calls → Kullanıcının giden çağrısı olup olmadığı (muhtemelen 0 veya 1).
- calls_outgoing_count → Yapılan toplam çağrı sayısı.
- calls_outgoing_spendings → Giden çağrılar için harcanan toplam ücret.
- calls_outgoing_duration → Giden çağrıların toplam süresi.
- calls_outgoing_spendings_max → Tek bir çağrı için maksimum harcama.
- calls_outgoing_duration_max → Tek bir çağrı için maksimum süre.
- calls_outgoing_inactive_days → Kullanıcının en son giden çağrısından bu yana geçen gün sayısı.

- calls_outgoing_to_onnet_count/spendings/duration/inactive_days → Aynı operatör içi giden çağrıların sayısı, harcaması, süresi, en son yapılan çağrıdan bu yana geçen gün.
- calls_outgoing_to_offnet_count/spendings/duration/inactive_days → Farklı operatörlere yapılan çağrılar için benzer bilgiler.
- calls_outgoing_to_abroad_count/spendings/duration/inactive_days → Yurtdışına yapılan çağrılar için bilgiler.

- user_has_outgoing_sms → Kullanıcının giden SMS gönderip göndermediği.
- sms_outgoing_count/spendings/spendings_max/inactive_days → Toplam gönderilen SMS sayısı, harcama, maksimum harcama ve en son SMS’ten bu yana geçen gün.
- sms_outgoing_to_onnet_count/spendings/inactive_days → Aynı operatör içi SMS’lerin bilgileri.
- sms_outgoing_to_offnet_count/spendings/inactive_days → Farklı operatörlere gönderilen SMS’ler.
- sms_outgoing_to_abroad_count/spendings/inactive_days → Yurtdışına gönderilen SMS’ler.
- sms_incoming_count/spendings → Gelen SMS sayısı ve maliyeti.
- sms_incoming_from_abroad_count/spendings → Yurtdışından gelen SMS’ler.

- user_use_gprs → Kullanıcının mobil internet kullanıp kullanmadığı.
- gprs_session_count → İnternet oturumlarının sayısı.
- gprs_usage → Kullanıcının toplam veri kullanımı (MB/GB cinsinden olabilir).
- gprs_spendings → İnternet kullanımına yapılan toplam harcama.
- gprs_inactive_days → En son mobil internet kullanımından bu yana geçen gün sayısı.

- user_does_reload → Kullanıcı bakiyesini dolduruyor mu? (0 veya 1)
- reloads_inactive_days → Kullanıcının en son bakiye yüklemesinden bu yana geçen gün.
- reloads_count/sum → Kullanıcının toplam bakiye yükleme sayısı ve yüklenen toplam tutar.

- last_100_reloads_count/sum → Son 100 bakiye yükleme işlemi için toplam yükleme sayısı ve miktarı.
- last_100_calls_outgoing_duration → Son 100 çağrının toplam süresi.
- last_100_calls_outgoing_to_onnet_duration → Son 100 aynı operatör içi çağrının süresi.
- last_100_calls_outgoing_to_offnet_duration → Son 100 farklı operatör içi çağrının süresi.
- last_100_calls_outgoing_to_abroad_duration → Son 100 yurtdışı çağrısının süresi.
- last_100_sms_outgoing_count → Son 100 SMS sayısı.
- last_100_sms_outgoing_to_onnet_count → Son 100 aynı operatöre giden SMS sayısı.
- last_100_sms_outgoing_to_offnet_count → Son 100 farklı operatöre giden SMS sayısı.
- last_100_sms_outgoing_to_abroad_count → Son 100 yurtdışı SMS sayısı.
- last_100_gprs_usage → Son 100 internet oturumu toplam veri kullanımı.
- churn → Kullanıcının operatörden ayrılıp ayrılmadığını gösteren etiket (0 = ayrılmadı, 1 = ayrıldı). Hedef değişken
