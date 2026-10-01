#!/usr/bin/env dotnet-script
#r "nuget: Microsoft.EntityFrameworkCore.Sqlite, 8.0.0"
#r "nuget: Microsoft.EntityFrameworkCore, 8.0.0"

using System.Security.Cryptography;
using Microsoft.Data.Sqlite;

var dbPath = Path.Combine(
    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
    "OContabil", "ocontabil.db");

Console.WriteLine($"DB: {dbPath}");

using var conn = new SqliteConnection($"Data Source={dbPath}");
conn.Open();

var cmd = conn.CreateCommand();
cmd.CommandText = "SELECT id, username, password_hash FROM users;";
using var reader = cmd.ExecuteReader();
while (reader.Read())
{
    Console.WriteLine($"ID={reader.GetInt32(0)} user={reader.GetString(1)} hash={reader.GetString(2)[..30]}...");
}
