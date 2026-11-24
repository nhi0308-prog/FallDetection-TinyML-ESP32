package com.example.fall_detection

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.database.DataSnapshot
import com.google.firebase.database.DatabaseError
import com.google.firebase.database.FirebaseDatabase
import com.google.firebase.database.ValueEventListener
import kotlinx.coroutines.delay
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private const val DATABASE_URL =
    "https://kfall-39eab-default-rtdb.asia-southeast1.firebasedatabase.app"

private const val DEVICE_ID = "kfall001"

private const val CHANNEL_SERVICE_ID = "kfall_service_channel"
private const val CHANNEL_FALL_ID = "kfall_fall_alert_channel"
private const val NOTIF_SERVICE_ID = 1001
private const val NOTIF_FALL_BASE_ID = 2000

// Bảng màu lấy cảm hứng từ logo KFall (Navy + Crimson) và giao diện Web sáng
private val Navy = Color(0xFF0F294A)
private val Blue = Color(0xFF1D5FA7)
private val BlueSoft = Color(0xFFEAF2FD)
private val SkyTop = Color(0xFFF3F8FE)
private val SkyBottom = Color(0xFFE2EEFA)
private val CardWhite = Color(0xFFFFFFFF)
private val Border = Color(0xFFC8DDF6)
private val Muted = Color(0xFF4F6B8A)
private val Crimson = Color(0xFFD61C4E)
private val CrimsonSoft = Color(0xFFFFEEF3)
private val Green = Color(0xFF0E7A5F)
private val Amber = Color(0xFFB46200)
private val AmberSoft = Color(0xFFFFF6E8)

private val KFallColors = lightColorScheme(
    primary = Blue,
    onPrimary = Color.White,
    primaryContainer = BlueSoft,
    onPrimaryContainer = Navy,
    background = SkyTop,
    onBackground = Navy,
    surface = CardWhite,
    onSurface = Navy,
    surfaceVariant = BlueSoft,
    onSurfaceVariant = Muted,
    outline = Border,
    error = Crimson,
    onError = Color.White,
    errorContainer = CrimsonSoft,
    onErrorContainer = Crimson
)

private data class DeviceReading(
    val bootId: String,
    val sequence: Long?,
    val status: String?,
    val siren: String?,
    val batteryPercent: Int?,
    val batteryAdcMv: Long?,
    val receivedAtMs: Long?
)

private data class AlertEntry(
    val id: String,
    val time: String,
    val message: String,
    val color: Color,
    val isFall: Boolean = false
)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            MaterialTheme(colorScheme = KFallColors) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = SkyTop
                ) {
                    KFallApp()
                }
            }
        }
    }
}

class FallMonitorService : Service() {
    private var database: FirebaseDatabase? = null
    private var latestListener: ValueEventListener? = null
    private var lastStatus: String? = null
    private var fallCounter = 0

    override fun onCreate() {
        super.onCreate()
        createNotificationChannels()
        startForeground(NOTIF_SERVICE_ID, buildServiceNotification())
        attachFirebaseListener()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForeground(NOTIF_SERVICE_ID, buildServiceNotification())
        if (latestListener == null) {
            attachFirebaseListener()
        }
        return START_STICKY
    }

    override fun onDestroy() {
        latestListener?.let { listener ->
            database?.getReference("devices/$DEVICE_ID/latest")
                ?.removeEventListener(listener)
        }
        latestListener = null
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun attachFirebaseListener() {
        val db = FirebaseDatabase.getInstance(DATABASE_URL)
        database = db
        val ref = db.getReference("devices/$DEVICE_ID/latest")

        val listener = object : ValueEventListener {
            override fun onDataChange(snapshot: DataSnapshot) {
                if (!snapshot.exists()) return

                val status = snapshot.child("status").getValue(String::class.java)
                val siren = snapshot.child("siren").getValue(String::class.java)
                val receivedAtMs = (snapshot.child("serverReceivedAtMs").value as? Number)?.toLong()

                if (status == "FALL" && lastStatus != "FALL") {
                    postFallNotification(
                        siren = siren,
                        receivedAtMs = receivedAtMs
                    )
                }
                lastStatus = status
            }

            override fun onCancelled(error: DatabaseError) {
                // Giữ service chạy để thử kết nối lại khi mạng ổn định
            }
        }
        latestListener = listener
        ref.addValueEventListener(listener)
    }

    private fun createNotificationChannels() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

            val serviceChannel = NotificationChannel(
                CHANNEL_SERVICE_ID,
                "Theo dõi thiết bị KFall",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Duy trì kết nối theo dõi thiết bị đeo khi chạy nền"
            }

            val fallChannel = NotificationChannel(
                CHANNEL_FALL_ID,
                "Cảnh báo ngã KFall",
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "Thông báo khẩn cấp khi phát hiện dấu hiệu ngã"
                enableVibration(true)
            }

            manager.createNotificationChannel(serviceChannel)
            manager.createNotificationChannel(fallChannel)
        }
    }

    private fun openAppPendingIntent(): PendingIntent {
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        return PendingIntent.getActivity(
            this,
            0,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
    }

    private fun buildServiceNotification(): Notification {
        return NotificationCompat.Builder(this, CHANNEL_SERVICE_ID)
            .setContentTitle("KFall đang theo dõi")
            .setContentText("Thiết bị $DEVICE_ID đang được giám sát")
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setContentIntent(openAppPendingIntent())
            .setOngoing(true)
            .setSilent(true)
            .build()
    }

    private fun postFallNotification(siren: String?, receivedAtMs: Long?) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            val granted = ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.POST_NOTIFICATIONS
            ) == PackageManager.PERMISSION_GRANTED
            if (!granted) return
        }

        val sirenLabel = when (siren) {
            "ON" -> "Đang kêu"
            "MUTED" -> "Đã tắt tiếng"
            "OFF" -> "Đang tắt"
            else -> "Chưa rõ"
        }
        val timeLabel = formatDateTime(receivedAtMs)

        val notification = NotificationCompat.Builder(this, CHANNEL_FALL_ID)
            .setContentTitle("Cảnh báo: Phát hiện ngã!")
            .setContentText("Thiết bị $DEVICE_ID · Còi: $sirenLabel · $timeLabel")
            .setStyle(
                NotificationCompat.BigTextStyle()
                    .bigText(
                        "Phát hiện dấu hiệu ngã trên thiết bị $DEVICE_ID lúc $timeLabel. " +
                                "Trạng thái còi: $sirenLabel. Hãy kiểm tra tình trạng người đeo ngay."
                    )
            )
            .setSmallIcon(android.R.drawable.ic_dialog_alert)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setCategory(NotificationCompat.CATEGORY_ALARM)
            .setAutoCancel(true)
            .setContentIntent(openAppPendingIntent())
            .build()

        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val notifId = NOTIF_FALL_BASE_ID + (fallCounter++ % 50)
        manager.notify(notifId, notification)
    }
}

@Composable
private fun KFallApp() {
    val context = LocalContext.current
    val auth = remember { FirebaseAuth.getInstance() }
    var currentUser by remember { mutableStateOf(auth.currentUser) }

    val notifPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { /* Kết quả quyền thông báo do người dùng chọn */ }

    DisposableEffect(auth) {
        val listener = FirebaseAuth.AuthStateListener {
            currentUser = it.currentUser
        }
        auth.addAuthStateListener(listener)

        onDispose {
            auth.removeAuthStateListener(listener)
        }
    }

    LaunchedEffect(currentUser?.uid) {
        if (currentUser != null) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                val hasPerm = ContextCompat.checkSelfPermission(
                    context,
                    Manifest.permission.POST_NOTIFICATIONS
                ) == PackageManager.PERMISSION_GRANTED
                if (!hasPerm) {
                    notifPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
                }
            }
            val serviceIntent = Intent(context, FallMonitorService::class.java)
            ContextCompat.startForegroundService(context, serviceIntent)
        } else {
            context.stopService(Intent(context, FallMonitorService::class.java))
        }
    }

    if (currentUser == null) {
        LoginScreen(auth)
    } else {
        key(currentUser!!.uid) {
            Dashboard(
                onSignOut = {
                    context.stopService(Intent(context, FallMonitorService::class.java))
                    auth.signOut()
                }
            )
        }
    }
}

@Composable
private fun LoginScreen(auth: FirebaseAuth) {
    var email by rememberSaveable { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var working by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }

    val fieldColors = OutlinedTextFieldDefaults.colors(
        focusedContainerColor = CardWhite,
        unfocusedContainerColor = CardWhite,
        focusedBorderColor = Blue,
        unfocusedBorderColor = Border,
        focusedLabelColor = Blue,
        unfocusedLabelColor = Muted,
        focusedTextColor = Navy,
        unfocusedTextColor = Navy,
        cursorColor = Blue
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                Brush.verticalGradient(
                    listOf(SkyTop, SkyBottom)
                )
            )
            .systemBarsPadding()
            .padding(24.dp),
        contentAlignment = Alignment.Center
    ) {
        Surface(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(24.dp),
            color = CardWhite,
            border = BorderStroke(1.dp, Border),
            shadowElevation = 2.dp
        ) {
            Column(
                modifier = Modifier.padding(24.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .size(12.dp)
                            .background(Crimson, CircleShape)
                    )
                    Text(
                        text = "KFall",
                        color = Navy,
                        style = MaterialTheme.typography.headlineLarge,
                        fontWeight = FontWeight.Bold
                    )
                }

                Text(
                    text = "Theo dõi thiết bị đeo của người thân",
                    color = Muted,
                    style = MaterialTheme.typography.bodyLarge
                )

                Spacer(Modifier.height(4.dp))

                OutlinedTextField(
                    value = email,
                    onValueChange = { email = it },
                    label = { Text("Email người thân") },
                    singleLine = true,
                    colors = fieldColors,
                    shape = RoundedCornerShape(14.dp),
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Email
                    ),
                    modifier = Modifier.fillMaxWidth()
                )

                OutlinedTextField(
                    value = password,
                    onValueChange = { password = it },
                    label = { Text("Mật khẩu") },
                    singleLine = true,
                    colors = fieldColors,
                    shape = RoundedCornerShape(14.dp),
                    visualTransformation = PasswordVisualTransformation(),
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Password
                    ),
                    modifier = Modifier.fillMaxWidth()
                )

                Spacer(Modifier.height(6.dp))

                Button(
                    onClick = {
                        working = true
                        error = null

                        auth.signInWithEmailAndPassword(
                            email.trim(),
                            password
                        ).addOnCompleteListener { result ->
                            working = false

                            if (result.isSuccessful) {
                                password = ""
                            } else {
                                error = result.exception?.localizedMessage
                                    ?: "Không thể đăng nhập"
                            }
                        }
                    },
                    enabled = !working &&
                            email.isNotBlank() &&
                            password.isNotBlank(),
                    shape = RoundedCornerShape(14.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Blue,
                        contentColor = Color.White,
                        disabledContainerColor = Border,
                        disabledContentColor = Muted
                    ),
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(50.dp)
                ) {
                    Text(
                        text = if (working) "Đang đăng nhập…" else "Đăng nhập",
                        fontWeight = FontWeight.SemiBold
                    )
                }

                error?.let {
                    Surface(
                        shape = RoundedCornerShape(12.dp),
                        color = CrimsonSoft,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = it,
                            color = Crimson,
                            style = MaterialTheme.typography.bodyMedium,
                            modifier = Modifier.padding(12.dp)
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun Dashboard(onSignOut: () -> Unit) {
    val database = remember {
        FirebaseDatabase.getInstance(DATABASE_URL)
    }

    var reading by remember { mutableStateOf<DeviceReading?>(null) }
    var readError by remember { mutableStateOf<String?>(null) }
    var serverOffsetMs by remember { mutableLongStateOf(0L) }
    var nowMs by remember { mutableLongStateOf(System.currentTimeMillis()) }
    var showFallDialog by remember { mutableStateOf(false) }

    val history = remember {
        mutableStateListOf<AlertEntry>()
    }

    LaunchedEffect(Unit) {
        while (true) {
            nowMs = System.currentTimeMillis()
            delay(1000)
        }
    }

    DisposableEffect(database) {
        val latestRef = database.getReference(
            "devices/$DEVICE_ID/latest"
        )

        val offsetRef = database.getReference(
            ".info/serverTimeOffset"
        )

        val latestListener = object : ValueEventListener {
            override fun onDataChange(snapshot: DataSnapshot) {
                readError = null

                if (!snapshot.exists()) {
                    reading = null
                    return
                }

                fun number(field: String): Long? {
                    return (snapshot.child(field).value as? Number)
                        ?.toLong()
                }

                val next = DeviceReading(
                    bootId = snapshot.child("bootId")
                        .getValue(String::class.java).orEmpty(),
                    sequence = number("sequence"),
                    status = snapshot.child("status")
                        .getValue(String::class.java),
                    siren = snapshot.child("siren")
                        .getValue(String::class.java),
                    batteryPercent = number("batteryPercent")
                        ?.toInt()
                        ?.takeIf { it in 0..100 },
                    batteryAdcMv = number("batteryAdcMv"),
                    receivedAtMs = number("serverReceivedAtMs")
                )

                val previous = reading

                if (
                    next.status == "FALL" &&
                    previous?.status != "FALL"
                ) {
                    val id = "${next.bootId}:${next.sequence}"

                    history.add(
                        0,
                        AlertEntry(
                            id = id,
                            time = formatTime(next.receivedAtMs),
                            message = "Phát hiện dấu hiệu ngã",
                            color = Crimson,
                            isFall = true
                        )
                    )
                }

                if (
                    next.siren == "MUTED" &&
                    previous?.siren != "MUTED"
                ) {
                    val id = "mute:${next.bootId}:${next.sequence}"

                    history.add(
                        0,
                        AlertEntry(
                            id = id,
                            time = formatTime(next.receivedAtMs),
                            message = "Đã tắt tiếng còi bằng VOL−",
                            color = Amber,
                            isFall = false
                        )
                    )
                }

                if (history.size > 100) {
                    history.removeAt(history.lastIndex)
                }

                reading = next
            }

            override fun onCancelled(error: DatabaseError) {
                readError = error.message
                reading = null
            }
        }

        val offsetListener = object : ValueEventListener {
            override fun onDataChange(snapshot: DataSnapshot) {
                serverOffsetMs =
                    (snapshot.value as? Number)?.toLong() ?: 0L
            }

            override fun onCancelled(error: DatabaseError) {
                // Đồng hồ điện thoại vẫn được dùng nếu không có offset.
            }
        }

        latestRef.addValueEventListener(latestListener)
        offsetRef.addValueEventListener(offsetListener)

        onDispose {
            latestRef.removeEventListener(latestListener)
            offsetRef.removeEventListener(offsetListener)
        }
    }

    val sample = reading
    val ageMs = sample?.receivedAtMs?.let {
        nowMs + serverOffsetMs - it
    }
    val fresh = ageMs != null &&
            ageMs >= -2000 &&
            ageMs <= 10_000

    val lastFallEvent = history.firstOrNull { it.isFall }
    val hasFallAlert = sample?.status == "FALL" || lastFallEvent != null

    val (statusText, statusDetail, statusColor) = when {
        readError != null -> Triple(
            "Không đọc được dữ liệu",
            readError.orEmpty(),
            Crimson
        )

        sample == null -> Triple(
            "Chờ thiết bị",
            "Chưa nhận được trạng thái từ Firebase.",
            Navy
        )

        !fresh -> Triple(
            "Dữ liệu cũ",
            "Lần cập nhật gần nhất: " +
                    formatDateTime(sample.receivedAtMs),
            Amber
        )

        sample.status == "FALL" -> Triple(
            "Phát hiện ngã",
            "Hãy kiểm tra tình trạng người đeo.",
            Crimson
        )

        sample.status == "NORMAL" -> Triple(
            "Bình thường",
            "Thiết bị đang theo dõi.",
            Green
        )

        sample.status == "WARMING_UP" -> Triple(
            "Đang khởi động",
            "Thiết bị đang chuẩn bị phát hiện ngã.",
            Blue
        )

        sample.status == "WAIT_MPU" -> Triple(
            "Lỗi cảm biến",
            "Thiết bị đang chờ cảm biến chuyển động.",
            Amber
        )

        sample.status == "AI_ERROR" -> Triple(
            "Lỗi phát hiện",
            "Thiết bị đang gặp lỗi xử lý AI.",
            Amber
        )

        else -> Triple(
            "Chưa xác định",
            "Trạng thái thiết bị: ${sample.status}",
            Amber
        )
    }

    val sirenText = when (sample?.siren) {
        "ON" -> "Đang kêu"
        "OFF" -> "Đang tắt"
        "MUTED" -> "Đã tắt tiếng"
        "WAIT" -> "Đang chờ"
        "ERROR" -> "Còi gặp lỗi"
        else -> "Chưa có dữ liệu"
    }

    val panelTint = when (statusColor) {
        Crimson -> CrimsonSoft
        Amber -> AmberSoft
        else -> CardWhite
    }

    if (showFallDialog) {
        val isActiveFall = sample?.status == "FALL"
        val dialogTitle = if (isActiveFall) {
            "Cảnh báo: Phát hiện ngã!"
        } else if (lastFallEvent != null) {
            "Thông báo ngã gần nhất"
        } else {
            "Thông báo ngã"
        }

        val dialogBody = when {
            isActiveFall ->
                "Thiết bị $DEVICE_ID đang báo trạng thái ngã (FALL). " +
                        "Còi cảnh báo: $sirenText. " +
                        "Thời gian ghi nhận: ${formatDateTime(sample.receivedAtMs)}. " +
                        "Hãy kiểm tra tình trạng người đeo ngay."

            lastFallEvent != null ->
                "Lần gần nhất phát hiện dấu hiệu ngã trong phiên lúc ${lastFallEvent.time}. " +
                        "Trạng thái hiện tại của thiết bị: $statusText."

            else ->
                "Hiện chưa ghi nhận cảnh báo ngã nào trên thiết bị $DEVICE_ID trong phiên này."
        }

        AlertDialog(
            onDismissRequest = { showFallDialog = false },
            containerColor = CardWhite,
            titleContentColor = if (hasFallAlert) Crimson else Navy,
            textContentColor = Muted,
            title = {
                Text(
                    text = dialogTitle,
                    fontWeight = FontWeight.Bold
                )
            },
            text = {
                Text(text = dialogBody)
            },
            confirmButton = {
                TextButton(
                    onClick = { showFallDialog = false }
                ) {
                    Text(
                        text = "Đã hiểu",
                        color = Blue,
                        fontWeight = FontWeight.SemiBold
                    )
                }
            }
        )
    }

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(
                Brush.verticalGradient(
                    listOf(SkyTop, SkyBottom)
                )
            )
            .systemBarsPadding(),
        contentPadding = PaddingValues(
            horizontal = 16.dp,
            vertical = 18.dp
        ),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        item {
            Surface(
                shape = RoundedCornerShape(22.dp),
                color = Navy,
                shadowElevation = 2.dp,
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(
                    modifier = Modifier.padding(20.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(10.dp)
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(10.dp)
                                    .background(Crimson, CircleShape)
                            )
                            Text(
                                "KFall",
                                color = Color.White,
                                style = MaterialTheme.typography.headlineMedium,
                                fontWeight = FontWeight.Bold
                            )
                        }

                        Box {
                            IconButton(
                                onClick = { showFallDialog = true }
                            ) {
                                val bellTint = if (hasFallAlert) {
                                    Color(0xFFFF8AA8)
                                } else {
                                    Color.White
                                }
                                Canvas(modifier = Modifier.size(24.dp)) {
                                    val w = size.width
                                    val h = size.height

                                    val bellPath = Path().apply {
                                        moveTo(w * 0.5f, h * 0.14f)
                                        cubicTo(
                                            w * 0.28f,
                                            h * 0.14f,
                                            w * 0.20f,
                                            h * 0.30f,
                                            w * 0.20f,
                                            h * 0.52f
                                        )
                                        lineTo(w * 0.12f, h * 0.72f)
                                        lineTo(w * 0.88f, h * 0.72f)
                                        lineTo(w * 0.80f, h * 0.52f)
                                        cubicTo(
                                            w * 0.80f,
                                            h * 0.30f,
                                            w * 0.72f,
                                            h * 0.14f,
                                            w * 0.5f,
                                            h * 0.14f
                                        )
                                        close()
                                    }
                                    drawPath(
                                        path = bellPath,
                                        color = bellTint,
                                        style = Stroke(
                                            width = w * 0.08f,
                                            cap = StrokeCap.Round,
                                            join = StrokeJoin.Round
                                        )
                                    )

                                    drawArc(
                                        color = bellTint,
                                        startAngle = 15f,
                                        sweepAngle = 150f,
                                        useCenter = false,
                                        topLeft = Offset(w * 0.36f, h * 0.74f),
                                        size = Size(w * 0.28f, h * 0.16f),
                                        style = Stroke(
                                            width = w * 0.08f,
                                            cap = StrokeCap.Round
                                        )
                                    )
                                }
                            }
                            if (hasFallAlert) {
                                Box(
                                    modifier = Modifier
                                        .align(Alignment.TopEnd)
                                        .offset(x = (-6).dp, y = 6.dp)
                                        .size(10.dp)
                                        .background(Crimson, CircleShape)
                                )
                            }
                        }
                    }

                    Text(
                        "Thiết bị người thân · $DEVICE_ID",
                        color = Color(0xFFD6E6FA)
                    )

                    Surface(
                        shape = RoundedCornerShape(999.dp),
                        color = if (fresh) Color(0xFF143A4A) else Color(0xFF3E2D16)
                    ) {
                        Text(
                            text = if (fresh) "● Đang nhận dữ liệu" else "● Chưa có dữ liệu mới",
                            color = if (fresh) Color(0xFF7CE8C6) else Color(0xFFFFD08A),
                            style = MaterialTheme.typography.labelLarge,
                            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)
                        )
                    }

                    OutlinedButton(
                        onClick = onSignOut,
                        border = BorderStroke(1.dp, Color(0xFF5D7FA8)),
                        colors = ButtonDefaults.outlinedButtonColors(
                            contentColor = Color.White
                        )
                    ) {
                        Text("Đăng xuất")
                    }
                }
            }
        }

        item {
            StatusPanel(
                title = "Trạng thái người đeo",
                accentColor = statusColor,
                tint = panelTint
            ) {
                Text(
                    statusText,
                    color = statusColor,
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold
                )
                Text(statusDetail, color = Muted)
            }
        }

        item {
            StatusPanel(
                title = "Còi cảnh báo",
                accentColor = if (sample?.siren == "ON") Crimson else Blue,
                tint = if (sample?.siren == "ON") CrimsonSoft else CardWhite
            ) {
                Text(
                    sirenText,
                    color = if (sample?.siren == "ON") Crimson else Blue,
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold
                )

                if (sample?.siren == "ON") {
                    Text(
                        "Nhấn VOL− trên thiết bị để tắt tiếng.",
                        color = Muted
                    )
                }

                if (sample?.siren == "MUTED") {
                    Text(
                        "Cảnh báo ngã vẫn cần được kiểm tra.",
                        color = Muted
                    )
                }
            }
        }

        item {
            StatusPanel(title = "Pin thiết bị") {
                val percent = sample?.batteryPercent
                val adcMv = sample?.batteryAdcMv

                when {
                    percent != null -> {
                        val batteryColor = when {
                            percent <= 15 -> Crimson
                            percent <= 30 -> Amber
                            else -> Blue
                        }
                        Text(
                            "$percent%",
                            color = batteryColor,
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold
                        )
                        LinearProgressIndicator(
                            progress = { percent / 100f },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(8.dp),
                            color = batteryColor,
                            trackColor = BlueSoft
                        )
                    }

                    adcMv != null -> {
                        Text(
                            "$adcMv mV tại chân ADC",
                            color = Blue,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            "Chưa hiệu chuẩn để tính phần trăm pin.",
                            color = Muted
                        )
                    }

                    else -> Text(
                        "Chưa có dữ liệu pin",
                        color = Muted
                    )
                }
            }
        }

        item {
            StatusPanel(title = "Lần cập nhật gần nhất") {
                Text(
                    formatDateTime(sample?.receivedAtMs),
                    color = Navy,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
            }
        }

        item {
            Text(
                "Nhật ký cảnh báo trong phiên app",
                color = Navy,
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold
            )
        }

        if (history.isEmpty()) {
            item {
                StatusPanel(title = "Nhật ký") {
                    Text(
                        "Chưa ghi nhận cảnh báo trong phiên này.",
                        color = Muted
                    )
                }
            }
        } else {
            items(
                items = history,
                key = { it.id }
            ) { event ->
                StatusPanel(
                    title = event.time,
                    accentColor = event.color,
                    tint = if (event.color == Crimson) CrimsonSoft else CardWhite
                ) {
                    Text(
                        event.message,
                        color = event.color,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }
    }
}

@Composable
private fun StatusPanel(
    title: String,
    accentColor: Color = Blue,
    tint: Color = CardWhite,
    content: @Composable ColumnScope.() -> Unit
) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(18.dp),
        color = tint,
        border = BorderStroke(
            width = 1.dp,
            color = if (tint == CardWhite) Border else accentColor.copy(alpha = 0.35f)
        ),
        shadowElevation = 1.dp
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(9.dp),
            content = {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .background(accentColor, CircleShape)
                    )
                    Text(
                        title,
                        color = Navy,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                }

                content()
            }
        )
    }
}

private fun formatTime(timestampMs: Long?): String {
    if (timestampMs == null) return "Chưa rõ giờ"

    return SimpleDateFormat(
        "HH:mm:ss",
        Locale.getDefault()
    ).format(Date(timestampMs))
}

private fun formatDateTime(timestampMs: Long?): String {
    if (timestampMs == null) return "Chưa có dữ liệu"

    return SimpleDateFormat(
        "dd/MM/yyyy HH:mm:ss",
        Locale.getDefault()
    ).format(Date(timestampMs))
}